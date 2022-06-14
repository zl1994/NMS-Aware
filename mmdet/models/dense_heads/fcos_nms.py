import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import Scale, normal_init, build_norm_layer, ConvModule
from mmcv.runner import force_fp32
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.models.utils import FFN, MultiheadAttention, build_transformer
from ortools.graph import pywrapgraph
from scipy.optimize import linear_sum_assignment

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

def network_flow_poto(cost_matrix, num_pos=3):
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    num_anchor = cost_matrix.shape[0]
    num_gt = cost_matrix.shape[1]
    start_nodes = [0]*num_gt
    end_nodes = [i for i in range(1, num_gt+1)]
    
    cost_matrix = cost_matrix.numpy().transpose(1,0)
    cost_list = []
    capacities = [num_pos]*num_gt
    # filter edges
    gt_index, anchor_index = np.where(cost_matrix!=-1.0)
    
    for i in range(len(gt_index)):
        start_nodes += [int(gt_index[i]+1)]
        end_nodes += [int(num_gt+1+anchor_index[i])]
        capacities += [1]
        cost_list += [int(1000000*(1-cost_matrix[gt_index[i], anchor_index[i]]))]
    
    capacities += [1]*num_anchor
    start_nodes = start_nodes + [k for k in range(num_gt+1, num_gt+num_anchor+1)]
    end_nodes = end_nodes + [num_gt+num_anchor+1]*num_anchor

    costs = [0]*num_gt+cost_list+[0]*num_anchor
    supplies = [num_gt*num_pos]+[0]*(num_gt+num_anchor)+[-num_gt*num_pos]
   
    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], costs[i])
    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    #min_cost_flow.Solve()
    min_cost_flow.SolveMaxFlowWithMinCost()
    #print(min_cost_flow.MaximumFlow())
    
    indice_anchor = []
    indice_gt = []

    source = 0
    sink = num_gt+num_anchor+1
    for arc in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(arc) != sink:
            if min_cost_flow.Flow(arc) > 0:
                indice_anchor.append(min_cost_flow.Head(arc))
                indice_gt.append(min_cost_flow.Tail(arc))
    return (np.array(indice_anchor)-1-num_gt, np.array(indice_gt)-1)

@HEADS.register_module()
class FCOSNMSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 poto_alpha=0.8,
                 num_pos=9,
                 nms_pre=1000,
                 stop_grad=True,
                 network_flow=True,
                 fuse_cls_reg=False,
                 bbox_refine=False,
                 dropout=0.0,
                 num_heads=8,
                 feedforward_channels=2048,
                 num_ffn_fcs=2,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=1,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 neighbor_num=20,
                 iou_mask=False,
                 exclude_self=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.poto_alpha = poto_alpha
        self.num_pos = num_pos
        self.stop_grad = stop_grad
        self.network_flow = network_flow
        self.fuse_cls_reg = fuse_cls_reg
        self.nms_pre = nms_pre
        self.num_heads = num_heads
        self.iou_mask=iou_mask
        self.neighbor_num=neighbor_num
        self.exclude_self=exclude_self
        self.bbox_refine=bbox_refine
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        
        self.attention2 = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm2 = build_norm_layer(dict(type='LN'), in_channels)[1]
        
        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
    
        self.ffn2 = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm2 = build_norm_layer(dict(type='LN'), in_channels)[1]
    
        self.res_cls_fc = nn.Linear(self.feat_channels, 1)
        if self.bbox_refine:
            self.res_reg_fc = nn.Linear(self.feat_channels, 4)
        
    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        super().init_weights()
    
    @torch.no_grad()
    def _matcher_poto(self,
                flat_points,
                cls_scores,
                bbox_preds,
                gt_bboxes,
                gt_labels,
                num_points_per_lvl,
                img_metas=None):
        """ 
        Performs the matching
        flat_points:[num_points, 2]
        cls_scores:[num_points, num_classes]
        bbox_preds:[num_points, 4]
        """  
        num_points = flat_points.shape[0]
        # We flatten to compute the cost matrices in a batch
        all_points = flat_points
        out_cls_scores = torch.sigmoid(cls_scores)
        out_bbox_preds = bbox_preds
        out_bbox = distance2bbox(all_points, out_bbox_preds)
        
        # Also concat the target labels and boxes
        tgt_ids = gt_labels
        tgt_bbox = gt_bboxes
        prob = out_cls_scores[:, tgt_ids]
        iou = bbox_overlaps(out_bbox, tgt_bbox)
        
        C = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha
        
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = all_points[:, 0], all_points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        
        # condition1: inside a `center bbox`
        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)
            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                                x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                                y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                                gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                                gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        C[~inside_gt_bbox_mask]=-1.0
        
        if self.network_flow:
            C = C.cpu()
            if self.num_pos==1:
                indices = linear_sum_assignment(C, maximize=True)
            else:
                indices = network_flow_poto(C, self.num_pos)
            return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]
        else:
            _, indices = torch.topk(C, self.num_pos, dim=0)
            anchor_index = torch.cat([indices[:, i] for i in range(indices.shape[1])])
            gt_index = torch.cat([torch.full((indices.shape[0],), i, dtype=torch.int64) for i in range(indices.shape[1])])
            return [anchor_index, gt_index]
    
    @torch.no_grad()
    def _matcher_poto_o2o(self,
                flat_points,
                cls_scores,
                bbox_preds,
                gt_bboxes,
                gt_labels,
                num_points_per_lvl,
                img_metas):
        """ 
        Performs the matching
        flat_points:[num_points, 2]
        cls_scores:[num_points, num_classes]
        bbox_preds:[num_points, 4]
        """  
        num_points = flat_points.shape[0]
        # We flatten to compute the cost matrices in a batch
        all_points = flat_points
        out_cls_scores = torch.sigmoid(cls_scores)
        out_bbox_preds = bbox_preds
        out_bbox = distance2bbox(all_points, out_bbox_preds)
        
        # Also concat the target labels and boxes
        tgt_ids = gt_labels
        tgt_bbox = gt_bboxes
        prob = out_cls_scores[:, tgt_ids]
        iou = bbox_overlaps(out_bbox, tgt_bbox)
        poto_alpha = 0.8
        C = prob ** (1 - poto_alpha) * iou ** poto_alpha
        
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = all_points[:, 0], all_points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        
        # condition1: inside a `center bbox`
        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)
            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                                x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                                y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                                gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                                gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        C[~inside_gt_bbox_mask]=-1.0
        C = C.cpu()
        indices = linear_sum_assignment(C, maximize=True)
        return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]
        
    @torch.no_grad()
    def _matcher_anchor(self,
                flat_points,
                cls_logits,
                bbox_preds):
        """ 
        Performs the matching
        """  
        nms_pre = self.nms_pre
        attention_nms_pre = self.nms_pre*2
        # We flatten to compute the cost matrices in a batch
        all_points = flat_points
        cls_scores = torch.sigmoid(cls_logits)
        bboxes = distance2bbox(all_points, bbox_preds)
        if nms_pre > 0 and cls_scores.shape[0] > nms_pre:
            max_scores, cls_inds = cls_scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            _, attention_topk_inds = max_scores.topk(attention_nms_pre)
            out_bboxes = bboxes[topk_inds, :]
            out_cls_logits = cls_logits[topk_inds, :]
            out_cls_scores = cls_scores[topk_inds, :]
            '''
            atten_out_bboxes = bboxes[attention_topk_inds, :]
            atten_out_cls_logits = cls_logits[attention_topk_inds, :]
            atten_out_cls_scores = cls_scores[attention_topk_inds, :]
            '''
        match_iou = bbox_overlaps(out_bboxes, out_bboxes)
        out_max_scores = max_scores[topk_inds]
        out_max_scores = out_max_scores.unsqueeze(dim=-1).repeat(1, out_max_scores.shape[0])
        edge_feature1 = out_cls_scores[:, cls_inds[topk_inds]]
        edge_feature2 = out_max_scores
        edge_feature3 = match_iou
        edge_feature = torch.cat([edge_feature1.unsqueeze(-1), edge_feature2.unsqueeze(-1), edge_feature3.unsqueeze(-1)], dim=-1)
        
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
                #print('iou:', match_iou[j, neighbor_inds[j]])
                #print('score:', edge_feature1[j, neighbor_inds[j]])
                if self.exclude_self:
                    A[j, j] = float('-inf') 
        

        
        A = A.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)
        return A.detach(), topk_inds
        #return A.detach(), neighbor_inds, edge_feature.detach(), topk_inds
    
    @torch.no_grad()
    def _matcher_o2o(self,
                flat_points,
                cls_scores,
                bbox_preds,
                gt_bboxes,
                gt_labels):
        """ 
        Performs the matching
        flat_points:[num_points, 2]
        cls_scores:[num_points, num_classes]
        bbox_preds:[num_points, 4]
        """  
        # We flatten to compute the cost matrices in a batch
        all_points = flat_points
        out_cls_scores = cls_scores
        out_bbox_preds = bbox_preds
        out_bbox = distance2bbox(all_points, out_bbox_preds)
        
        # Also concat the target labels and boxes
        tgt_ids = gt_labels
        tgt_bbox = gt_bboxes
        prob = out_cls_scores[:, tgt_ids]
        iou = bbox_overlaps(out_bbox, tgt_bbox)

        C = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha
        
        C = C.cpu()
        indices = linear_sum_assignment(C, maximize=True)
        return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        # stop the grad
        if self.stop_grad:
            if self.fuse_cls_reg:
                node_feat = torch.cat((cls_feat.detach(), reg_feat.detach()), dim=1)
            else:
                node_feat = reg_feat.detach()
        else:
            if self.fuse_cls_reg:
                node_feat = torch.cat((cls_feat, reg_feat), dim=1)
            else:
                node_feat = reg_feat

        return cls_score, bbox_pred, node_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'node_feats'))
    def loss(self,
             cls_scores,
             bbox_preds,
             node_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        
        num_points_per_lvl = [points.shape[0] for points in all_level_points]

        num_points_per_lvl_list = []
        all_points_list = []
        cls_scores_list = []
        bbox_preds_list = []
        node_feats_list = []
        strides_list = []

        num_imgs = cls_scores[0].size(0)

        for i in range(num_imgs): 
            num_points_per_lvl_list.append(num_points_per_lvl)
            all_points_list.append(torch.cat([points for points in all_level_points]))
            cls_scores_list.append(torch.cat([cls_score[i].permute(1, 2, 0).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]))
            #cls_scores_list.append(torch.cat([cls_score[i].permute(1, 2, 0).reshape(-1, 1) for cls_score in cls_scores]))
            node_feats_list.append(torch.cat([node_feat[i].permute(1, 2, 0).reshape(-1, 256) for node_feat in node_feats]))
            # norm_on_bbox
            bbox_preds_list.append(torch.cat([(bbox_pred[i]*stride).permute(1, 2, 0).reshape(-1, 4) for bbox_pred, stride in zip(bbox_preds, self.strides)]))
            strides_list.append(torch.cat([bbox_pred.new_full(((bbox_pred[i].permute(1, 2, 0).reshape(-1, 4)).shape[0], 1), stride) for bbox_pred, stride in zip(bbox_preds, self.strides)]))
        
        # ------------------------------------one-to-many loss--------------------------------------------------------------------------
        # one-to-many lable assignment
        indices_list = multi_apply(
                self._matcher_poto,
                all_points_list,
                cls_scores_list,
                bbox_preds_list,
                gt_bboxes,
                gt_labels,
                num_points_per_lvl_list,
                img_metas)
        
        pos_inds_list = []
        pos_gt_bboxes_list = []
        pos_gt_lables_list = []
       
        for i in range(num_imgs):
            # clone() is important
            pos_inds = indices_list[0][i].clone()
            gt_inds = indices_list[1][i].clone()
            pos_inds_list.append(pos_inds)
            pos_gt_bboxes_list.append(gt_bboxes[i][gt_inds, :])
            pos_gt_lables_list.append(gt_labels[i][gt_inds])

        cnt = 0
        for i in range(1, num_imgs):
            cnt += all_points_list[i-1].shape[0]
            pos_inds_list[i] += cnt

        #pos_inds_list[1] += all_points_list[0].shape[0]
        pos_inds = torch.cat(pos_inds_list)
        pos_gt_bboxes = torch.cat(pos_gt_bboxes_list)
        pos_gt_lables = torch.cat(pos_gt_lables_list)
        
        all_points = torch.cat(all_points_list)
        all_cls_scores = torch.cat(cls_scores_list)
        all_bbox_preds = torch.cat(bbox_preds_list)
        
        all_labels = all_points.new_full((all_points.shape[0], ), self.num_classes, dtype=torch.long)
        all_labels[pos_inds] = pos_gt_lables
        
        
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        pos_points = all_points[pos_inds]
        pos_bbox_preds = all_bbox_preds[pos_inds]
        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        # cls loss
        #loss_cls = self.loss_cls(all_cls_scores, all_labels, avg_factor=num_pos)
        
        # VFL 
        score = all_cls_scores.new_zeros((all_cls_scores.shape[0]))
        score[pos_inds] = bbox_overlaps(
                pos_decoded_bbox_preds.detach(),
                pos_gt_bboxes, is_aligned=True)
        
        cls_weight = all_cls_scores.new_ones((all_cls_scores.shape[0]))
        cls_weight[pos_inds] = 1.0
        #cls_weight[pos_inds] = (1+score[pos_inds])**0.9
        #weight_denorm = max(reduce_mean(cls_weight[pos_inds].sum().detach()), 1e-6)
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            all_cls_scores, (all_labels, score),
            weight=cls_weight.detach(),
            avg_factor=num_pos)
        
        # reg loss
        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_gt_bboxes,
                avg_factor=num_pos)
            
        else:
            loss_bbox = pos_bbox_preds.sum()
        
        # ------------------------------------one-to-one loss---------------------------------------------------------------------------
        # get top1000 and its matching matrix
        out = multi_apply(
            self._matcher_anchor,
            all_points_list,
            cls_scores_list,
            bbox_preds_list)
        '''
        top_neighbor_inds_list = out[1]
        top_neighbor_inds_list[1] += top_neighbor_inds_list[0].shape[0]
        top_neighbor_inds = torch.cat(top_neighbor_inds_list, dim=0)
        '''
        inds_list = out[-1]
        top_cls_logits_list = []
        top_reg_preds_list = []
        edge_feature_list = []
        top_node_feature_list = []
        top_node_cls_feature_list = []
        repeat_node_feats_list = []
        A_list = []
        top_strides_list = []
        agg_points_list = []
        for i in range(num_imgs):
            A_list.append(out[0][i])
            # top_cls_logits:Nx80
            top_cls_logits = cls_scores_list[i][inds_list[i], :]
            top_reg_preds = bbox_preds_list[i][inds_list[i], :]
            # stop the grad
            top_cls_logits = top_cls_logits.detach()
            top_reg_preds = top_reg_preds.detach()
            
            agg_points_list.append(all_points_list[i][inds_list[i], :])
            top_cls_logits_list.append(top_cls_logits)
            top_reg_preds_list.append(top_reg_preds)
            node_feats = node_feats_list[i][inds_list[i], :]
            strides = strides_list[i][inds_list[i]]
            top_strides_list.append(strides.unsqueeze(dim=0))
            top_node_feature_list.append(node_feats.unsqueeze(dim=0))
        A = torch.cat(A_list, dim=0)
        
        # top_node_features：BxNxC
        top_node_features = torch.cat(top_node_feature_list, dim=0)
        top_strides = torch.cat(top_strides_list, dim=0)
        
        nms_embedding_feat = top_node_features
        B, N, C = nms_embedding_feat.shape
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.attention_norm(self.attention(nms_embedding_feat, attn_mask=A))
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
      
        nms_embedding_feat = self.ffn_norm(self.ffn(nms_embedding_feat))
    
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.attention_norm2(self.attention2(nms_embedding_feat, attn_mask=A))
        nms_embedding_feat = nms_embedding_feat.permute(1, 0, 2)
        nms_embedding_feat = self.ffn_norm2(self.ffn2(nms_embedding_feat))
        
        if self.bbox_refine:
            res_reg_preds = self.res_reg_fc(nms_embedding_feat)

        res_cls_logits = self.res_cls_fc(nms_embedding_feat)
        
        
       
        agg_cls_scores_list = []
        for i in range(num_imgs):
            agg_cls_scores_list.append(torch.sigmoid(top_cls_logits_list[i])*torch.sigmoid(res_cls_logits[i]))
     
        '''
        agg_cls_scores_list = [torch.sigmoid(ori_cls_logits1)*torch.sigmoid(res_cls_logits[0]), 
                               torch.sigmoid(ori_cls_logits2)*torch.sigmoid(res_cls_logits[1])]
        '''
        if self.bbox_refine:
            ori_reg_preds1 = top_reg_preds_list[0]
            ori_reg_preds2 = top_reg_preds_list[1]
            bbox_preds_refine = res_reg_preds.float().exp()
            agg_bbox_preds_list = [ori_reg_preds1*bbox_preds_refine[0],
                                   ori_reg_preds2*bbox_preds_refine[1]]
        else:
            agg_bbox_preds_list = []
            for i in range(num_imgs): 
                agg_bbox_preds = bbox_preds_list[i][inds_list[i], :]
                agg_bbox_preds_list.append(agg_bbox_preds)
        
        # one-to-one lable assignment
        o2o_indices_list = multi_apply(
            self._matcher_o2o,
            agg_points_list,
            agg_cls_scores_list,
            agg_bbox_preds_list,
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
            cnt += agg_points_list[i-1].shape[0]
            o2o_pos_inds_list[i] += cnt
        #o2o_pos_inds_list[1] += agg_points_list[0].shape[0]
        o2o_pos_inds = torch.cat(o2o_pos_inds_list)
        o2o_pos_gt_bboxes = torch.cat(o2o_pos_gt_bboxes_list)
        o2o_pos_gt_lables = torch.cat(o2o_pos_gt_lables_list)
        all_agg_cls_scores = torch.cat(agg_cls_scores_list)
        all_agg_bbox_preds = torch.cat(agg_bbox_preds_list)
        all_agg_points = torch.cat(agg_points_list)
        
        all_mask_labels = all_agg_points.new_full((all_agg_points.shape[0], ), self.num_classes, dtype=torch.long)
        all_mask_labels[o2o_pos_inds] = o2o_pos_gt_lables

        valid_idxs = all_mask_labels >= 0
        pos_idxs = (all_mask_labels >= 0) & (all_mask_labels != self.num_classes)
        num_pos = pos_idxs.sum()
        gt_classes_target = torch.zeros_like(all_agg_cls_scores)
        gt_classes_target[pos_idxs, all_mask_labels[pos_idxs]] = 1

        loss_o2o_cls = focal_loss_jit(
            all_agg_cls_scores[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=0.5,
            gamma=1.0,
            reduction="sum",
        ) / max(1.0, num_pos)
        
        if self.bbox_refine:
            # reg loss
            agg_pos_bbox_preds = all_agg_bbox_preds[pos_idxs]
            if len(pos_inds) > 0:
                agg_pos_points = all_agg_points[pos_idxs]
                agg_pos_decoded_bbox_preds = distance2bbox(agg_pos_points, agg_pos_bbox_preds)
                loss_o2o_reg = self.loss_bbox(
                    agg_pos_decoded_bbox_preds,
                    o2o_pos_gt_bboxes,
                    avg_factor=num_pos)
            else:
                loss_o2o_reg = agg_pos_bbox_preds.sum()

            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,loss_o2o_cls=loss_o2o_cls, loss_o2o_reg=loss_o2o_reg)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,loss_o2o_cls=loss_o2o_cls)
        
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'node_feats'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   node_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   gt_bboxes=None, 
                   gt_labels=None):
        '''
        Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        '''
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        

        num_points_per_lvl = [points.shape[0] for points in mlvl_points]
        result_list = []
        for img_id in range(len(img_metas)):
            all_points = torch.cat([points for points in mlvl_points])
            strides = torch.cat([bbox_pred.new_full(((bbox_pred[img_id].permute(1, 2, 0).reshape(-1, 4)).shape[0], 1), stride) for bbox_pred, stride in zip(bbox_preds, self.strides)])
            cls_scores = torch.cat([cls_score[img_id].permute(1, 2, 0).reshape(-1, self.cls_out_channels) for cls_score in cls_scores])
            bbox_preds = torch.cat([bbox_pred[img_id].permute(1, 2, 0).reshape(-1, 4) for bbox_pred in bbox_preds])
            node_feats = torch.cat([node_feat[img_id].permute(1, 2, 0).reshape(-1, 256) for node_feat in node_feats])
            
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                num_points_per_lvl, cls_scores, bbox_preds, node_feats, strides,
                all_points, img_shape, featmap_sizes, scale_factor, cfg, rescale, gt_bboxes=gt_bboxes[img_id][0], 
                   gt_labels=gt_labels[img_id][0], with_nms=False)
            
            result_list.append(det_bboxes)
  
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    node_feats,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True,
                    gt_bboxes=None,
                    gt_labels=None):

        gt_bboxes = gt_bboxes[0]
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        strides = [8.0, 16.0, 32.0, 64.0, 128.0]
        dis_list = []
        iou_list = []
        max_scores_list = []
        bbox_pred_list = []
        node_feat_list = []
        dot_similarity_list = []
        cnt=1
        for cls_score, bbox_pred, node_feat, points, stride in zip(
                cls_scores, bbox_preds, node_feats, mlvl_points, strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            
            h=bbox_pred.shape[-2]
            w=bbox_pred.shape[-1]
            origin_bbox_pred = bbox_pred
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            node_feat = node_feat.permute(0, 2, 3, 1)
            node_feat_list.append(node_feat[0])
            # Always keep topk op for dynamic input in onnx
            '''
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)
              
                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
            '''
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)

            max_scores,_ = scores.max(dim=-1)
            max_scores = max_scores.reshape(-1, h, w)
            max_scores_list.append(max_scores[0].detach())
            iou = bbox_overlaps(gt_bboxes, bboxes.squeeze(dim=0))
            taget_gt_bboxes = gt_bboxes.unsqueeze(dim=1)
            dis = torch.abs(taget_gt_bboxes-bboxes)
            dis = dis.reshape(-1, h, w, 4)
            iou = iou.reshape(-1, h, w)
            dis_list.append(dis)
            iou_list.append(iou)
            bbox_pred_list.append(bbox_pred[0].reshape(h, w, 4))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
    
        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = batch_mlvl_scores.max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
                -1, 1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]

        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img)
            
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results, dis_list, iou_list, bbox_pred_list, max_scores_list, node_feat_list
    
    def _get_bboxes_single(self,
                           num_points_per_lvl,
                           cls_scores,
                           bbox_preds,
                           node_feats,
                           strides,
                           points,
                           img_shape,
                           featmap_sizes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           gt_bboxes=None, 
                           gt_labels=None,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)
        '''
        if gt_bboxes.shape[0]:
            inds_list = self._matcher_poto(points,
                                            cls_scores,
                                            bbox_preds,
                                            gt_bboxes,
                                            gt_labels,
                                            num_points_per_lvl)
            pos_inds = inds_list[0].clone()
            gt_inds = inds_list[1].clone()

            _, num_counts=torch.unique(gt_inds, return_counts=True)
            
            
            level_list = []
            scores_list = []
            scores = torch.sigmoid(cls_scores)
            for i in range(pos_inds.numel()):
                lvl_begin = 0
                ind = pos_inds[i]
                scores_list.append(scores[ind, gt_labels[gt_inds[i]]])
                for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                    lvl_end = lvl_begin + num_points_lvl
                    if ind < lvl_end and ind >= lvl_begin:
                        lvl = lvl_idx
                        break
                    lvl_begin = lvl_end
                level_list.append(lvl)
            levels = torch.tensor(level_list)
            scores = torch.tensor(scores_list)
            
            level_list  = []
            scores_list = []
            begin = 0
            for _, num_count in enumerate(num_counts):
                end = begin+num_count
                level = levels[begin:end]
                score = scores[begin:end]
                _, num_level_counts = torch.unique(level, return_counts=True)
                
                level_instance_list = []
                score_instance_list = []
                instance_begin = 0
                for _, num_level_count in enumerate(num_level_counts):
                    instance_end = instance_begin+num_level_count
                    #level_instance_list.append(level[instance_begin:instance_end])
                    score_instance_list.append(score[instance_begin:instance_end].max())
                    instance_begin = instance_end
                
                level_instance_list = num_level_counts
                level_list.append(level_instance_list)
                scores_list.append(score_instance_list)
                begin = end

            print('level_list:', level_list)
            print('scores_list:', scores_list)
        '''
        num_topk = min(cfg.max_per_img, points.shape[0])
        if rescale:
            gt_bboxes /= gt_bboxes.new_tensor(scale_factor)
            
        scores = torch.sigmoid(cls_scores)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, cls_indices = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            tmp_points = points[topk_inds, :]
            tmp_bbox_preds = bbox_preds[topk_inds, :]
            tmp_scores = scores[topk_inds, :]
        
        max_scores = max_scores[topk_inds]
        max_scores = max_scores.unsqueeze(dim=-1).repeat(1, max_scores.shape[0])
        top1000_points = tmp_points
        top1000_logits = cls_scores[topk_inds, :]
        top1000_node_feats = node_feats[topk_inds, :]
    
        top1000_strides = strides[topk_inds]
        top1000_scores = tmp_scores
        top1000_bboxes = distance2bbox(tmp_points, tmp_bbox_preds, max_shape=img_shape)
        
        match_iou = bbox_overlaps(top1000_bboxes, top1000_bboxes)
        edge_feature1 = top1000_scores[:, cls_indices[topk_inds]]
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
        
        if with_nms:
            scores = cls_scores.sigmoid_()
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, cls_indices = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_preds = bbox_preds[topk_inds, :]
                scores = scores[topk_inds, :]
            
            bboxes = distance2bbox(points, bbox_preds, max_shape=img_shape)
            if rescale:
                bboxes /= bboxes.new_tensor(scale_factor)
            '''
            max_scores, _ = scores.max(dim=1)
            if gt_bboxes.shape[0]!=0:
                all_IoU = bbox_overlaps(gt_bboxes, bboxes)
                thresh = all_IoU > 0.5
                filter_IoU = torch.where(all_IoU>0.5, all_IoU, all_IoU.new_zeros(all_IoU.shape))
                max_IoU,_ = filter_IoU.max(dim=1)
                filter_IoU /= max_IoU.unsqueeze(dim=1)
                top_IoU, _  = torch.topk(filter_IoU, 5, dim=1)
                top_IoU = torch.sum(top_IoU, dim=1)
                gt_count = torch.sum(thresh, dim=1, dtype=torch.float)

                anchor_IoU, index = all_IoU.max(dim=0)
            else:
                anchor_IoU = None
                index = None
                max_scores = None
                #top_IoU = None
                #gt_count = None
            '''
            padding = scores.new_zeros(scores.shape[0], 1)
            scores = torch.cat([scores, padding], dim=1)
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)

            return det_bboxes, det_labels#, top_IoU, gt_count
        else:
            
            top1000_node_feats = top1000_node_feats.unsqueeze(dim=0)
            nms_embedding_feat = top1000_node_feats
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
                res_reg_preds = self.res_reg_fc(nms_embedding_feat)
                bbox_preds_refine = res_reg_preds.float().exp()
                agg_bbox_preds = tmp_bbox_preds*bbox_preds_refine.squeeze(dim=0)
                top1000_bboxes = distance2bbox(tmp_points, agg_bbox_preds, max_shape=img_shape)
        
            if rescale:
                top1000_bboxes /= top1000_bboxes.new_tensor(scale_factor)

            #cls_scores = torch.sqrt(torch.sigmoid(top1000_logits)*torch.sigmoid(res_cls_logits))
            cls_scores = torch.sigmoid(top1000_logits)*torch.sigmoid(res_cls_logits)
            
            vis_cls_scores = torch.sigmoid(top1000_logits)
            
            cls_scores = cls_scores.flatten()            
            # Keep top k top scoring indices only.
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = cls_scores.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > cfg.score_thr
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            
            bboxes = top1000_bboxes[shift_idxs]
            points = top1000_points[shift_idxs]
            
            det_bboxes = torch.cat([bboxes, predicted_prob.unsqueeze(dim=-1)], -1)
            det_labels = classes_idxs
            return det_bboxes, det_labels

    def score_voting(self, det_bboxes, det_labels, mlvl_bboxes,
                     mlvl_nms_scores, mlvl_centerness, score_thr):
        """Implementation of score voting method works on each remaining boxes
        after NMS procedure.

        Args:
            det_bboxes (Tensor): Remaining boxes after NMS procedure,
                with shape (k, 5), each dimension means
                (x1, y1, x2, y2, score).
            det_labels (Tensor): The label of remaining boxes, with shape
                (k, 1),Labels are 0-based.
            mlvl_bboxes (Tensor): All boxes before the NMS procedure,
                with shape (num_anchors,4).
            mlvl_nms_scores (Tensor): The scores of all boxes which is used
                in the NMS procedure, with shape (num_anchors, num_class)
            mlvl_iou_preds (Tensot): The predictions of IOU of all boxes
                before the NMS procedure, with shape (num_anchors, 1)
            score_thr (float): The score threshold of bboxes.

        Returns:
            tuple: Usually returns a tuple containing voting results.

                - det_bboxes_voted (Tensor): Remaining boxes after
                    score voting procedure, with shape (k, 5), each
                    dimension means (x1, y1, x2, y2, score).
                - det_labels_voted (Tensor): Label of remaining bboxes
                    after voting, with shape (num_anchors,).
        """
        candidate_mask = mlvl_nms_scores > score_thr
        candidate_mask_nozeros = candidate_mask.nonzero()
        candidate_inds = candidate_mask_nozeros[:, 0]
        candidate_labels = candidate_mask_nozeros[:, 1]
        candidate_bboxes = mlvl_bboxes[candidate_inds]
        candidate_scores = mlvl_nms_scores[candidate_mask]
        candidate_centerness = mlvl_centerness[candidate_inds]
        det_bboxes_voted = []
        det_labels_voted = []
        for cls in range(self.cls_out_channels):
            candidate_cls_mask = candidate_labels == cls
            if not candidate_cls_mask.any():
                continue
            candidate_cls_scores = candidate_scores[candidate_cls_mask]
            candidate_cls_bboxes = candidate_bboxes[candidate_cls_mask]
            candidate_cls_centerness = candidate_centerness[candidate_cls_mask]
            det_cls_mask = det_labels == cls
            det_cls_bboxes = det_bboxes[det_cls_mask].view(
                -1, det_bboxes.size(-1))
            det_candidate_ious = bbox_overlaps(det_cls_bboxes[:, :4],
                                               candidate_cls_bboxes)
            for det_ind in range(len(det_cls_bboxes)):
                single_det_ious = det_candidate_ious[det_ind]
                #print(single_det_ious.shape, candidate_cls_centerness.shape)
                pos_ious_mask = (single_det_ious > 0.9) & (candidate_cls_centerness>0.5)
                '''
                pos_ious = single_det_ious[pos_ious_mask]
                pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                pos_scores = candidate_cls_scores[pos_ious_mask]
                pis = (torch.exp(-(1 - pos_ious)**2 / 0.025) * pos_scores)[:, None]
                voted_box = torch.sum(pis * pos_bboxes, dim=0) / torch.sum(pis, dim=0)
                '''
                if pos_ious_mask.any():
                    pos_ious = single_det_ious[pos_ious_mask]
                    pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                    voted_box = pos_bboxes.mean(dim=0)
                else:
                    pos_ious_mask = (single_det_ious > 0.01)
                    pos_ious = single_det_ious[pos_ious_mask]
                    pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                    pos_scores = candidate_cls_scores[pos_ious_mask]
                    pis = (torch.exp(-(1 - pos_ious)**2 / 0.025) * pos_scores)[:, None]
                    voted_box = torch.sum(pis * pos_bboxes, dim=0) / torch.sum(pis, dim=0)
                
                
                voted_score = det_cls_bboxes[det_ind][-1:][None, :]
                det_bboxes_voted.append(
                    torch.cat((voted_box[None, :], voted_score), dim=1))
                det_labels_voted.append(cls)

        det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
        det_labels_voted = det_labels.new_tensor(det_labels_voted)
        return det_bboxes_voted, det_labels_voted

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
