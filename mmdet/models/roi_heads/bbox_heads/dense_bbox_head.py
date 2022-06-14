import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, sidedistance2bbox, bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmcv.cnn import ConvModule

@HEADS.register_module()
class DensePredHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 side_separate=False,
                 std=0.1,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(DensePredHead, self).__init__()
        assert with_cls or with_reg
        self.side_separate=side_separate
        self.std = std
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False


        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        
        # add reg specific branch
        if not self.side_separate:
            self.reg_convs, self.reg_fcs, self.reg_last_dim = \
                self._add_conv_fc_branch(
                    self.num_reg_convs, self.num_reg_fcs, self.in_channels)
        else:
            self.reg_convs, self.reg_fcs, self.reg_last_dim = \
                self._add_conv_fc_branch(0, self.num_reg_fcs, self.in_channels)
            self.reg_convs_x1 = nn.ModuleList()
            self.reg_convs_y1 = nn.ModuleList()
            self.reg_convs_x2 = nn.ModuleList()
            self.reg_convs_y2 = nn.ModuleList()
            if self.num_reg_convs > 0:
                for i in range(self.num_reg_convs):
                    conv_in_channels = (
                        self.in_channels if i == 0 else self.conv_out_channels)
                    self.reg_convs_x1.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg))
                    self.reg_convs_y1.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg))
                    self.reg_convs_x2.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg))
                    self.reg_convs_y2.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg))

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            '''
            self.conv_reg_x = ConvModule(
                        self.conv_out_channels,
                        self.conv_out_channels,
                        (1, 7),
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
            self.conv_reg_y = ConvModule(
                        self.conv_out_channels,
                        self.conv_out_channels,
                        (7, 1),
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
            '''
            self.fc_reg_x1 = nn.Linear(self.conv_out_channels, 1)
            self.fc_reg_y1 = nn.Linear(self.conv_out_channels, 1)
            self.fc_reg_x2 = nn.Linear(self.conv_out_channels, 1)
            self.fc_reg_y2 = nn.Linear(self.conv_out_channels, 1)

            self.fc_side_x1 = nn.Linear(self.conv_out_channels, 1)
            self.fc_side_y1 = nn.Linear(self.conv_out_channels, 1)
            self.fc_side_x2 = nn.Linear(self.conv_out_channels, 1)
            self.fc_side_y2 = nn.Linear(self.conv_out_channels, 1)
            #self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
            self.fc_side = nn.Linear(self.conv_out_channels, 4)
       
    
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg_x1.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_x1.bias, 0)
            nn.init.normal_(self.fc_reg_y1.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_y1.bias, 0)
            nn.init.normal_(self.fc_reg_x2.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_x2.bias, 0)
            nn.init.normal_(self.fc_reg_y2.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_y2.bias, 0)
            nn.init.normal_(self.fc_side.weight, 0, 0.001)
            nn.init.constant_(self.fc_side.bias, 0)

            nn.init.normal_(self.fc_side_x1.weight, 0, 0.001)
            nn.init.constant_(self.fc_side_x1.bias, 0)
            nn.init.normal_(self.fc_side_y1.weight, 0, 0.001)
            nn.init.constant_(self.fc_side_y1.bias, 0)
            nn.init.normal_(self.fc_side_x2.weight, 0, 0.001)
            nn.init.constant_(self.fc_side_x2.bias, 0)
            nn.init.normal_(self.fc_side_y2.weight, 0, 0.001)
            nn.init.constant_(self.fc_side_y2.bias, 0)
            
            nn.init.normal_(self.fc_side.weight, 0, 0.001)
            nn.init.constant_(self.fc_side.bias, 0)
            '''
            nn.init.normal_(self.conv_reg_x.weight, 0, 0.001)
            nn.init.constant_(self.conv_reg_x.bias, 0)
            nn.init.normal_(self.conv_reg_y.weight, 0, 0.001)
            nn.init.constant_(self.conv_reg_y.bias, 0)
            '''
    def forward(self, x):
        # shared part
        #x_reg = reg_bbox_feats
        x_reg = x
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        if not self.side_separate:
            for conv in self.reg_convs:
                x_reg = conv(x_reg)
                
            kernel_size = x_reg.shape[-1]
            '''
            top_bottom = self.conv_reg_x(x_reg)
            left_right = self.conv_reg_y(x_reg)
            '''
            top_bottom = F.max_pool2d(x_reg, (1, kernel_size), stride=1)
            left_right = F.max_pool2d(x_reg, (kernel_size, 1), stride=1)
            
            top    = top_bottom[:, :, 0, :].squeeze(dim=-1)
            bottom = top_bottom[:, :, -1, :].squeeze(dim=-1)
            left   = left_right[:, :, :, 0].squeeze(dim=-1)
            right  = left_right[:, :, :, -1].squeeze(dim=-1)
            '''
            x_roi = self.avg_pool(x_reg)
            x_roi = x_roi[:, :, 0, 0]
            left = x_roi
            top = x_roi
            right = x_roi
            bottom = x_roi
            '''
        else:
            x_reg_left = x_reg
            x_reg_top = x_reg
            x_reg_right = x_reg
            x_reg_bottom = x_reg
            for conv in self.reg_convs_x1:
                x_reg_left = conv(x_reg_left)
            for conv in self.reg_convs_y1:
                x_reg_top = conv(x_reg_top)
            for conv in self.reg_convs_x2:
                x_reg_right = conv(x_reg_right)
            for conv in self.reg_convs_y2:
                x_reg_bottom = conv(x_reg_bottom)
            
            kernel_size = x_reg.shape[-1]
            top    = F.max_pool2d(x_reg_top, (1, kernel_size), stride=1)
            bottom = F.max_pool2d(x_reg_bottom, (1, kernel_size), stride=1)
            left   = F.max_pool2d(x_reg_left, (kernel_size, 1), stride=1)
            right   = F.max_pool2d(x_reg_right, (kernel_size, 1), stride=1)

            top    = top[:, :, 0, :].squeeze(dim=-1)
            bottom = bottom[:, :, -1, :].squeeze(dim=-1)
            left   = left[:, :, :, 0].squeeze(dim=-1)
            right  = right[:, :, :, -1].squeeze(dim=-1)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred_x1 = self.fc_reg_x1(left) if self.with_reg else None
        bbox_pred_y1 = self.fc_reg_y1(top) if self.with_reg else None
        bbox_pred_x2 = self.fc_reg_x2(right) if self.with_reg else None
        bbox_pred_y2 = self.fc_reg_y2(bottom) if self.with_reg else None
        
        side_x1 = self.fc_side_x1(left) if self.with_reg else None
        side_y1 = self.fc_side_y1(top) if self.with_reg else None
        side_x2 = self.fc_side_x2(right) if self.with_reg else None
        side_y2 = self.fc_side_y2(bottom) if self.with_reg else None
        
        bbox_pred = torch.cat([bbox_pred_x1, bbox_pred_y1, bbox_pred_x2, bbox_pred_y2], dim=-1)
        side_pred = torch.cat([side_x1, side_y1, side_x2, side_y2], dim=-1)
        
        return cls_score, bbox_pred, side_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'side_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             side_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            w = (rois[..., 3]-rois[..., 1]).clamp(min=1e-6)
            h = (rois[..., 4]-rois[..., 2]).clamp(min=1e-6)
            
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            num_pos = pos_inds.sum()
            num_pos = max(num_pos, 1)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    #bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    output_size = 7
                    spacing_w = 0.5 * w / float(output_size)
                    spacing_h = 0.5 * h / float(output_size)
                    
                    
                    decode_bboxes = bbox_pred#.clone().detach()
                    decode_bboxes[..., 0] = decode_bboxes[..., 0] * w * self.std
                    decode_bboxes[..., 1] = decode_bboxes[..., 1] * h * self.std
                    decode_bboxes[..., 2] = decode_bboxes[..., 2] * w * self.std
                    decode_bboxes[..., 3] = decode_bboxes[..., 3] * h * self.std
                    decode_bboxes = sidedistance2bbox(rois[..., 1:], decode_bboxes)  
                    '''
                    origin_bbox_targets = bbox_targets
                    
                    gt_left = ((bbox_targets[..., 0]-rois[..., 1])/w)/self.std
                    gt_top = ((bbox_targets[..., 1]-rois[..., 2])/h)/self.std
                    gt_right = ((bbox_targets[..., 2]-rois[..., 3])/w)/self.std
                    gt_bottom = ((bbox_targets[..., 3]-rois[..., 4])/h)/self.std
                    bbox_targets = torch.stack([gt_left, gt_top, gt_right, gt_bottom], dim=-1)
                    
                    w = (decode_bboxes[..., 2]-decode_bboxes[..., 0]).clamp(min=1e-6)
                    h = (decode_bboxes[..., 3]-decode_bboxes[..., 1]).clamp(min=1e-6)
                    iou = bbox_overlaps(decode_bboxes, origin_bbox_targets, is_aligned=True)
                    filter_pos_inds = (iou>0.5)
                    
                    gt_side_left = ((origin_bbox_targets[..., 0]-decode_bboxes[..., 0])/w)/0.05
                    gt_side_top = ((origin_bbox_targets[..., 1]-decode_bboxes[..., 1])/h)/0.05
                    gt_side_right = ((origin_bbox_targets[..., 2]-decode_bboxes[..., 2])/w)/0.05
                    gt_side_bottom = ((origin_bbox_targets[..., 3]-decode_bboxes[..., 3])/h)/0.05
                    side_targets = torch.stack([gt_side_left, gt_side_top, gt_side_right, gt_side_bottom], dim=-1)
                    '''
                #side_pos_inds=filter_pos_inds & pos_inds
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                    #pos_side_pred = side_pred.view(
                    #    side_pred.size(0), 4)[side_pos_inds.type(torch.bool)]
                    
                    pos_decode_bboxes = decode_bboxes.view(
                        decode_bboxes.size(0), 4)[pos_inds.type(torch.bool)]
                    
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                
                losses['loss_bbox'] = self.loss_bbox(
                                    pos_decode_bboxes,
                                    bbox_targets[pos_inds.type(torch.bool)],
                                    avg_factor=num_pos)
                '''
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            
                losses['loss_side'] = self.loss_bbox(
                    pos_side_pred,
                    side_targets[side_pos_inds.type(torch.bool)],
                    bbox_weights[side_pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                '''
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   side_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   gt_bboxes=None,
                   gt_labels=None):
        """Transform network output for a batch into bbox predictions.

        If the input rois has batch dimension, the function would be in
        `batch_mode` and return is a tuple[list[Tensor], list[Tensor]],
        otherwise, the return is a tuple[Tensor, Tensor].

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5)
               or (B, num_boxes, 5)
            cls_score (list[Tensor] or Tensor): Box scores for
               each scale level, each is a 4D-tensor, the channel number is
               num_points * num_classes.
            bbox_pred (Tensor, optional): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_classes * 4.
            img_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]], optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If rois shape is (B, num_boxes, 4), then
                the max_shape should be a Sequence[Sequence[int]]
                and the length of max_shape should also be B.
            scale_factor (tuple[ndarray] or ndarray): Scale factor of the
               image arange as (w_scale, h_scale, w_scale, h_scale). In
               `batch_mode`, the scale_factor shape is tuple[ndarray].
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[list[Tensor], list[Tensor]] or tuple[Tensor, Tensor]:
                If the input has a batch dimension, the return value is
                a tuple of the list. The first list contains the boxes of
                the corresponding image in a batch, each tensor has the
                shape (num_boxes, 5) and last dimension 5 represent
                (tl_x, tl_y, br_x, br_y, score). Each Tensor in the second
                list is the labels with shape (num_boxes, ). The length of
                both lists should be equal to batch_size. Otherwise return
                value is a tuple of two tensors, the first tensor is the
                boxes with scores, the second tensor is the labels, both
                have the same shape as the first case.
        """
       
        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]
    
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
         
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None

        batch_mode = True
        if rois.ndim == 2:
            # e.g. AugTest, Cascade R-CNN, HTC, SCNet...
            batch_mode = False

            # add batch dimension
            if scores is not None:
                scores = scores.unsqueeze(0)
            if bbox_pred is not None:
                bbox_pred = bbox_pred.unsqueeze(0)
            rois = rois.unsqueeze(0)
        
        if bbox_pred is not None:
            w = (rois[..., 3]-rois[..., 1]).clamp(min=1e-6)
            h = (rois[..., 4]-rois[..., 2]).clamp(min=1e-6)
            output_size = 7
            spacing_w = 0.5 * w / float(output_size)
            spacing_h = 0.5 * h / float(output_size)
          
            ori_bbox_pred = bbox_pred.clone()
            bbox_pred[..., 0] = bbox_pred[..., 0] * w * self.std#+spacing_w
            bbox_pred[..., 1] = bbox_pred[..., 1] * h * self.std#+spacing_h
            bbox_pred[..., 2] = bbox_pred[..., 2] * w * self.std#-spacing_w
            bbox_pred[..., 3] = bbox_pred[..., 3] * h * self.std#-spacing_h
            bboxes = sidedistance2bbox(rois[..., 1:], bbox_pred, max_shape=img_shape)
            '''
            bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
            '''
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)
        
        if gt_bboxes.shape[0]!=0:
            iou = bbox_overlaps(rois[0][:, 1:], gt_bboxes)
            val, inds = iou.max(dim=1)
            filter_inds = (val>0.5).nonzero(as_tuple=True)[0]
            target_gt_bboxes = gt_bboxes[inds]
            '''
            gt_left = (torch.abs(target_gt_bboxes[:, 0]-rois[0][:, 1])/w.squeeze(dim=0))/self.std
            gt_top = (torch.abs(target_gt_bboxes[:, 1]-rois[0][:, 2])/h.squeeze(dim=0))/self.std
            gt_right = (torch.abs(target_gt_bboxes[:, 2]-rois[0][:, 3])/w.squeeze(dim=0))/self.std
            gt_bottom = (torch.abs(target_gt_bboxes[:, 3]-rois[0][:, 4])/h.squeeze(dim=0))/self.std
            side_distance = torch.stack([gt_left, gt_top, gt_right, gt_bottom], dim=-1)
            #side_distance = torch.abs(rois[0][:, 1:]-target_gt_bboxes)
            side_confids = torch.abs(ori_bbox_pred[0])
            '''
            
            w = (bboxes[..., 2]-bboxes[..., 0]).clamp(min=1e-6)
            h = (bboxes[..., 3]-bboxes[..., 1]).clamp(min=1e-6)
          
            gt_left = (torch.abs(target_gt_bboxes[:, 0]-bboxes[0][:, 0])/w.squeeze(dim=0))/self.std
            gt_top = (torch.abs(target_gt_bboxes[:, 1]-bboxes[0][:, 1])/h.squeeze(dim=0))/self.std
            gt_right = (torch.abs(target_gt_bboxes[:, 2]-bboxes[0][:, 2])/w.squeeze(dim=0))/self.std
            gt_bottom = (torch.abs(target_gt_bboxes[:, 3]-bboxes[0][:, 3])/h.squeeze(dim=0))/self.std
            side_distance = torch.stack([gt_left, gt_top, gt_right, gt_bottom], dim=-1)
            side_confids = torch.abs(side_pred[0])
            

            side_distance = side_distance[filter_inds]
            side_confids = side_confids[filter_inds]
        else:
            side_distance = None
            side_confids = None
            filter_inds = None
        
        out_bboxes = bboxes.clone()
      
        if rescale and bboxes.size(-2) > 0:
            if not isinstance(scale_factor, tuple):
                scale_factor = tuple([scale_factor])
            # B, 1, bboxes.size(-1)
            scale_factor = bboxes.new_tensor(scale_factor).unsqueeze(1).repeat(
                1, 1,
                bboxes.size(-1) // 4)
            bboxes /= scale_factor
        
        det_bboxes = []
        det_labels = []
        for (bbox, score) in zip(bboxes, scores):
            if cfg is not None:
                det_bbox, det_label = multiclass_nms(bbox, score,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
            else:
                det_bbox, det_label = bbox, score
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if not batch_mode:
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
        return out_bboxes[0], filter_inds, det_bboxes, det_labels, side_distance, side_confids

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
