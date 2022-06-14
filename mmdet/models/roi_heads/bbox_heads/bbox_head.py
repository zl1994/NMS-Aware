import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmcv.cnn import Scale, normal_init, build_norm_layer, ConvModule
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, reduce_mean
from mmdet.models.utils import build_linear_layer, FFN, MultiheadAttention, build_transformer
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.iou_calculators import bbox_overlaps

INF = 1e8

def focal_loss(
    probs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    ce_loss = F.binary_cross_entropy(
        probs, targets, reduction="none"
    )
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

focal_loss_jit = torch.jit.script(focal_loss)  # type: torch.jit.ScriptModule

@HEADS.register_module()
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
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
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
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

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
            self.fc_side = nn.Linear(in_channels, 4)
        self.debug_imgs = None

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
            nn.init.normal_(self.fc_side.weight, 0, 0.001)
            nn.init.constant_(self.fc_side.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

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
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
    
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
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
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

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
        return det_bboxes, det_labels

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


@HEADS.register_module()
class BBoxHeadOur(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 o2m_num=9,
                 poto_alpha=0.8,
                 neighbor_num=20,
                 iou_mask=True,
                 bbox_refine=False,
                 exclude_self=True,
                 dropout=0.0,
                 num_heads=8,
                 feedforward_channels=2048,
                 num_ffn_fcs=2,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 stop_grad=True,
                 o2o_channels=256,
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
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_refine_bbox=dict(type='GIoULoss', loss_weight=2.0)):
        super(BBoxHeadOur, self).__init__()
        assert with_cls or with_reg
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
        # param of o2o
        self.o2o_channels = o2o_channels
        self.o2m_num      = o2m_num
        self.poto_alpha   = poto_alpha
        self.stop_grad    = stop_grad
        self.num_heads    = num_heads
        self.iou_mask     = iou_mask
        self.neighbor_num = neighbor_num
        self.exclude_self = exclude_self
        self.bbox_refine  = bbox_refine

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_refine_bbox = build_loss(loss_refine_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        self.res_fc = nn.Linear(1024, self.o2o_channels)
        self.attention = MultiheadAttention(self.o2o_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), self.o2o_channels)[1]
        self.attention2 = MultiheadAttention(self.o2o_channels, num_heads, dropout)
        self.attention_norm2 = build_norm_layer(dict(type='LN'), self.o2o_channels)[1]
        self.ffn = FFN(
            self.o2o_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), self.o2o_channels)[1]
        self.ffn2 = FFN(
            self.o2o_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm2 = build_norm_layer(dict(type='LN'), self.o2o_channels)[1]

        self.res_cls_fc = nn.Linear(self.o2o_channels, 1)
        if self.bbox_refine:
            self.res_reg_fc = nn.Linear(self.o2o_channels, 4)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
        if self.bbox_refine:
            nn.init.normal_(self.res_reg_fc.weight, 0, 0.001)
            nn.init.constant_(self.res_reg_fc.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
    
    @torch.no_grad()
    def _matcher_roi(self,
                rois,
                cls_scores,
                bbox_preds):
        """ 
        Performs the matching
        """  
        # We flatten to compute the cost matrices in a batch
        #cls_scores = torch.sigmoid(cls_logits)
        bboxes = self.bbox_coder.decode(rois, bbox_preds)
        max_scores, cls_inds = cls_scores.max(dim=1)

        match_iou = bbox_overlaps(bboxes, bboxes)
        edge_feature1 = cls_scores[:, cls_inds]
        edge_feature3 = match_iou
        
        
        if self.iou_mask:
            adjacency_matrix = match_iou
            A = adjacency_matrix.new_full(adjacency_matrix.shape, float('-inf'))
            for j in range(adjacency_matrix.shape[0]):
                A[j, j] = 1e-6
            A[adjacency_matrix>0.5]=0
        else:
            adjacency_matrix = edge_feature1 ** (1 - self.poto_alpha) * match_iou ** self.poto_alpha
            A = adjacency_matrix.new_full(adjacency_matrix.shape, float('-inf'))
            _, neighbor_inds = adjacency_matrix.topk(self.neighbor_num, dim=-1)
            for j in range(adjacency_matrix.shape[0]):
                A[j, neighbor_inds[j]] = 0
                if self.exclude_self:
                    A[j, j] = float('-inf') 
        
        A = A.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)
        return [A.detach()]
    
    @torch.no_grad()
    def _matcher_o2o(self,
                rois,
                cls_scores,
                bbox_preds,
                agg_bbox_pred,
                gt_bboxes,
                gt_labels,
                num=1):
        # We flatten to compute the cost matrices in a batch
        out_cls_scores = cls_scores
        out_bbox = self.bbox_coder.decode(rois, bbox_preds)
        #out_bbox = self.bbox_coder.decode(rois, agg_bbox_pred)
        #out_bbox += agg_bbox_pred
        # Also concat the target labels and boxes
        tgt_ids = gt_labels
        tgt_bbox = gt_bboxes
        prob = out_cls_scores[:, tgt_ids]
        iou = bbox_overlaps(out_bbox, tgt_bbox)
        
        C = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha
        C = C.cpu()
        if num>1:
            iou_mask = (iou>0.3)
            C[~iou_mask]=-1.0
            _, indices = torch.topk(C, num, dim=0)
            '''
            mean_iou = iou[indices, [i for i in range(iou.shape[1])]]
            print(mean_iou.mean(dim=0))
            '''
            anchor_index = torch.cat([indices[:, i] for i in range(indices.shape[1])])
            gt_index = torch.cat([torch.full((indices.shape[0],), i, dtype=torch.int64) for i in range(indices.shape[1])])
            return [anchor_index, gt_index]
        else:
            iou_mask = (iou>0.3)
            C[~iou_mask]=-1.0
            indices = linear_sum_assignment(C, maximize=True)
            return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]
    
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

    @force_fp32(apply_to=('all_cls_score', 'all_bbox_pred', 'all_bbox_feature'))
    def loss(self,
             all_cls_score,
             all_bbox_pred,
             all_bbox_feature,
             rois,
             rois_list,
             pos_inds_list,
             neg_inds_list,
             num_gt_list,
             gt_bboxes,
             gt_labels,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        num_imgs = len(rois_list)
        batch_size = [num_gt_list[i]+rois_list[i].shape[0] for i in range(num_imgs)]
        cnt = 0
        sample_inds_list = []
        sample_inds_list.append(torch.cat([pos_inds_list[0], neg_inds_list[0]]))
        for i in range(1, num_imgs):
            assign_pos_inds = pos_inds_list[i].clone()
            assign_neg_inds = neg_inds_list[i].clone()
            cnt += batch_size[i-1]
            assign_pos_inds += cnt
            assign_neg_inds += cnt
            sample_inds_list.append(torch.cat([assign_pos_inds, assign_neg_inds]))
        
        sample_inds = torch.cat(sample_inds_list)

        cls_score = all_cls_score[sample_inds]
        bbox_pred = all_bbox_pred[sample_inds]

        if self.stop_grad:
            all_bbox_feature = all_bbox_feature.detach()

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
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
    
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        
        # ------------------------------------one-to-one loss---------------------------------------------------------------------------
        # get top1000 and its matching matrix
        cls_score_list = []
        bbox_pred_list = []
        bbox_feature_list = []
        start = 0
        end   = 0
        for i in range(num_imgs):
            start += num_gt_list[i]
            end += batch_size[i]
            o2m_cls_score = all_cls_score.clone().detach()
            o2m_bbox_pred = all_bbox_pred.clone().detach()
            scores = F.softmax(o2m_cls_score[start:end], dim=-1)
            cls_score_list.append(scores[:, :-1])
            bbox_pred_list.append(o2m_bbox_pred[start:end])
            tmp_bbox_feature = all_bbox_feature[start:end]
            tmp_bbox_feature = tmp_bbox_feature.view(tmp_bbox_feature.shape[0], -1)
            bbox_feature_list.append(tmp_bbox_feature.unsqueeze(dim=0))
            start = end
        
        out = multi_apply(
            self._matcher_roi,
            rois_list,
            cls_score_list,
            bbox_pred_list)
        
        A_list = []
        for i in range(num_imgs):
            A_list.append(out[0][i])
        A = torch.cat(A_list, dim=0)
        
        # top_node_features：BxNxC
        top_node_features = torch.cat(bbox_feature_list, dim=0)
        nms_embedding_feat = top_node_features
        nms_embedding_feat = self.relu(self.res_fc(nms_embedding_feat))

        B, N, C = nms_embedding_feat.shape
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.attention_norm(self.attention(nms_embedding_feat, attn_mask=A))
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.ffn_norm(self.ffn(nms_embedding_feat))
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.attention_norm2(self.attention2(nms_embedding_feat, attn_mask=A))
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.ffn_norm2(self.ffn2(nms_embedding_feat))
        
        
        res_cls_logits = self.res_cls_fc(nms_embedding_feat)
        if self.bbox_refine:
            res_bbox_preds = self.res_reg_fc(nms_embedding_feat)
        
        agg_cls_scores_list = []
        res_bbox_pred_list = []
        for i in range(num_imgs):
            agg_cls_scores_list.append(cls_score_list[i]*torch.sigmoid(res_cls_logits[i]))
            #agg_cls_scores_list.append(cls_score_list[i]*res_cls_logits[i])
            res_bbox_pred_list.append(res_bbox_preds[i])
        
        # one-to-one lable assignment
        o2o_indices_list = multi_apply(
            self._matcher_o2o,
            rois_list,
            agg_cls_scores_list,
            bbox_pred_list,
            res_bbox_pred_list,
            gt_bboxes,
            gt_labels)
        
        o2o_pos_inds_list = []
        o2o_pos_gt_bboxes_list = []
        o2o_pos_gt_lables_list = []
        for i in range(num_imgs):
            o2o_pos_inds_list.append(o2o_indices_list[0][i])
            o2o_pos_gt_bboxes_list.append(gt_bboxes[i][o2o_indices_list[1][i], :])
            o2o_pos_gt_lables_list.append(gt_labels[i][o2o_indices_list[1][i]])
            
        cnt = 0 
        for i in range(1, num_imgs):
            cnt += rois_list[i-1].shape[0]
            o2o_pos_inds_list[i] += cnt
        
        all_rois = torch.cat(rois_list)
        o2o_pos_inds = torch.cat(o2o_pos_inds_list)
        o2o_pos_gt_bboxes = torch.cat(o2o_pos_gt_bboxes_list)
        o2o_pos_gt_lables = torch.cat(o2o_pos_gt_lables_list)
        all_agg_cls_scores = torch.cat(agg_cls_scores_list)
        all_res_bbox_preds = torch.cat(res_bbox_pred_list)
        o2m_bbox_preds = torch.cat(bbox_pred_list)
        all_mask_labels = all_agg_cls_scores.new_full((all_agg_cls_scores.shape[0], ), self.num_classes, dtype=torch.long)
        all_mask_labels[o2o_pos_inds] = o2o_pos_gt_lables
        
        pos_idxs = (all_mask_labels >= 0) & (all_mask_labels != self.num_classes)
        num_pos = torch.tensor(len(o2o_pos_inds), dtype=torch.float, device=cls_score.device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        #----------------------------------------------------
        gt_classes_target = torch.zeros_like(all_agg_cls_scores)
        gt_classes_target[pos_idxs, all_mask_labels[pos_idxs]] = 1
        loss_o2o_cls = focal_loss_jit(
            all_agg_cls_scores,
            gt_classes_target,
            alpha=0.5,
            gamma=1.0,
            reduction="sum",
        ) / max(1.0, num_pos)
        #-----------------------------------------------
        losses['loss_o2o_cls'] = loss_o2o_cls

        if self.bbox_refine:
            # reg loss
            all_decode_preds = self.bbox_coder.decode(all_rois, all_res_bbox_preds)
            #all_decode_preds = self.bbox_coder.decode(all_rois, o2m_bbox_preds)
            #all_decode_preds = all_decode_preds + all_res_bbox_preds
            agg_pos_bbox_preds = all_decode_preds[pos_idxs]
            if len(pos_inds) > 0:
                loss_o2o_reg = self.loss_refine_bbox(
                    agg_pos_bbox_preds,
                    o2o_pos_gt_bboxes,
                    avg_factor=num_pos)
            else:
                loss_o2o_reg = all_res_bbox_preds.sum()
            losses['loss_o2o_reg'] = loss_o2o_reg
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_feats'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   bbox_feats,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   with_nms=False):
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
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

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
        for (bbox, score, bbox_feat) in zip(bboxes, scores, bbox_feats):
            if with_nms:
                if cfg is not None:
                    det_bbox, det_label = multiclass_nms(bbox, score,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
                else:
                    det_bbox, det_label = bbox, score
            else:
                match_iou = bbox_overlaps(bbox, bbox)
                A = match_iou.new_full(match_iou.shape, float('-inf'))
                for j in range(match_iou.shape[0]):
                    A[j, j] = 1e-6
                A[match_iou>0.5]=0
                A = A.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)

                top1000_node_feats = bbox_feat.unsqueeze(dim=0)
                nms_embedding_feat = top1000_node_feats
                B, N, C = nms_embedding_feat.shape
                nms_embedding_feat = self.relu(self.res_fc(nms_embedding_feat))

                nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
                nms_embedding_feat = self.attention_norm(self.attention(nms_embedding_feat, attn_mask=A))
                nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
                nms_embedding_feat = self.ffn_norm(self.ffn(nms_embedding_feat))
                
                nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
                nms_embedding_feat = self.attention_norm2(self.attention2(nms_embedding_feat, attn_mask=A))
                nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
                nms_embedding_feat = self.ffn_norm2(self.ffn2(nms_embedding_feat))
                
                res_cls_logits = self.res_cls_fc(nms_embedding_feat)
              
                cls_scores = score[:, :-1]*torch.sigmoid(res_cls_logits)
                if self.bbox_refine:
                    res_bbox_preds = self.res_reg_fc(nms_embedding_feat)
                    bbox = self.bbox_coder.decode(rois[..., 1:], res_bbox_preds, max_shape=img_shape)
                    #bboxes = bbox + res_bbox_preds
                if rescale and bbox.size(0) > 0:
                    scale_factor = bbox.new_tensor(scale_factor)
                    bbox /= scale_factor
                
                bbox = bbox[0]
                cls_scores = cls_scores.flatten()            
                # Keep top k top scoring indices only.
                # torch.sort is actually faster than .topk (at least on GPUs)
                predicted_prob, topk_idxs = cls_scores.sort(descending=True)
                predicted_prob = predicted_prob[:cfg.max_per_img]
                topk_idxs = topk_idxs[:cfg.max_per_img]
                # filter out the proposals with low confidence score
                keep_idxs = predicted_prob > cfg.score_thr
                predicted_prob = predicted_prob[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]
                shift_idxs = topk_idxs // self.num_classes
                classes_idxs = topk_idxs % self.num_classes                
                bbox = bbox[shift_idxs]

                det_bbox = torch.cat([bbox, predicted_prob.unsqueeze(dim=-1)], -1)
                det_label = classes_idxs

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if not batch_mode:
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
        return det_bboxes, det_labels

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
