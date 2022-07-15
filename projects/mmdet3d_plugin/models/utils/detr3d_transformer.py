import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
import warnings
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks.transformer import BaseTransformerLayer

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
    # transform inter-queries to 3d points coordinate
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
    # reg_branches在detr3d_head处定义
    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encod ing
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        # 一个作为position_embedding, 一个作为content_embedding
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)

        # pos shape:[bs, num_query, embed_dims],在bs个batch上的query保持一直
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        # query shape:[bs, num_query,embed_dims]
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # points shape:[bs, num_query, 3]
        reference_points = self.reference_points(query_pos)
        # transform to [0,1]
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        # (num_dec_layers, bs, num_query, embed_dims),(bs, num_queries, 4),(num_dec_layers, bs, num_query, 4)
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    # 通过cfg传入的参数中，args=(), return_intermediate=True, kwargs=dict(...), **kwargs会自动捕捉除了指定的形参之外的所有赋值参数
    def __init__(self, *args, return_intermediate=False, **kwargs):
        # args:num_layers = 6, **kwargs:transformerlayers_cfg
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    # mlvl_feats:[bs, embed_dims, h, w]
    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape (bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        # layer_id, layer
        # 这里的layer是包含了multihead-atten和cross-atten以及各项配置的完整block
        # 参考坐标回归等在两个block之间，也就是layer之间
        # reference points初始化为query初始映射
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)
            # 更新reference points
            if reg_branches is not None:
                #  预测的是偏移
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3
                # inverse_sigmoid(reference) + bias -> x,y
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                new_reference_points = new_reference_points.sigmoid()
                # 不同points之间独立，不进行梯度共享和数值共享
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        # after stack: [6, num_query, bs, embed_dims], [num_layers=6, bs, num_query, 3]
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

# 在Detr3DCrossAtten之前，先经过了MultiheadAttention
@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=1,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # 输入的是query和reference points
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        
        # shape:(B, N, C, H, W), [bs, num_query, 3]
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])

        # set nan = 0,可能部分sample_feature为nan，由于其在图像之外
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        # 对每个camera中投影在之外的点进行mask
        # attention_weights:[bs, 1, num_query, num_cams, 1, n_l] * [bs, 1, n_q, n_cams, 1, 1] -> [bs, 1, n_q, n_c, 1, n_l]
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights # [B, C, num_query, num_cam,  1, num_level] * [B, 1, num_query, num_cam,  1, num_level]
        output = output.sum(-1).sum(-1).sum(-1) # sum后缩减三个维度:shape:[bs, c, num_query],聚合每个query所指向的所有点的所有层所有相机的特征值
        output = output.permute(2, 0, 1) # [num_query, bs, c]
        
        output = self.output_proj(output) # sample_feature -> 256dims 方便相加
        # reference pts 经过encoder再次用于下一个layer的reference pts，在喂入encoder之前，先经过inverse_sigmoid还原坐标
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat

# 特征采样部分， Input queries from different level. Each element has shape [bs, embed_dims, h, w] 也就是[4, bs, embed_dims, h, w]
def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    # lidar2img:3D坐标以lidar为中心，求出3D点到img的转换关系也就是求出lidar到img的转换关系
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    # N = 6，referrence_points:[bs, num_query, 3]
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    #  还原归一化后的[x,y,z]
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points [bs, num_query, 3] -> [bs, num_query, 4]
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    # num_cam = 6
    num_cam = lidar2img.size(1)
    # from [b,1,num_query,4] to [b,num_cam,num_query, 4, 1],将相同的query投影到每个camera中，因为每个query对应的是lidar坐标系的三维点
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # shape:[b, num_cam, num_query, 4, 4]
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    # project 3d -> 2d
    # shape:[b, num_cam, num_query, 4]，计算得到每个三维点在每个相机的投影
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5 # eps作为无穷小数
    # [bs, n_c, n_q]
    mask = (reference_points_cam[..., 2:3] > eps)

    # 避免除以0的问题, reference_points_cam[..., 2:3]指的是深度,要获得z*[x,y,1,1] = cam_intrinsics[4x4] * [x,y,z,1] -> z*[x,y,1,1] / z -> [x,y,1,1]
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    # 归一化[0,1]
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    # 确保reference在[-1, 1]之间,原来在[0,1]之间
    reference_points_cam = (reference_points_cam - 0.5) * 2

    # 对所有不在grid内的点,也就是投影在某个cam之外的点进行mask
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5) # [bs, 1, n_q, n_c, 1, 1]
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    # 对四个特征层分别求出线性插值后的feature，其中N为num_query, [4, bs, embed_dims, h, w]
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size() # (num_key, bs, embed_dims)
        # N=num_cam
        feat = feat.view(B*N, C, H, W)
        # [b,num_cam,num_query,2] -> [b, num_cam, num_query, 1, 2]
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        # F.grid_sample return:[b*n,c,num_query,1]每个query对应着一个grid采样(bilinear incorparation)后返回的值
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # b,c,n_q,n,1
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    # [b,n,c,num_query,len(mlvl_feats)]
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats)) 
    return reference_points_3d, sampled_feats, mask
