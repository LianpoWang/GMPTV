import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer import FeatureTransformer3D, FeatureTransformer3D_PT, GCL
from .matching import global_correlation_softmax_3d, SelfCorrelationSoftmax3D
from .backbone import PointNet, DGCNN, MLP

def B_N_xyz2pxo(input,max_points):
    b, n, xyz = input.size()
    if b>1:
        n_o, count = [ max_points ],max_points
        tmp = [input[0]]
        for i in range(1, b):
            count += max_points
            n_o.append(count)
            tmp.append(input[i])
        n_o = torch.cuda.IntTensor(n_o)
        coord = torch.cat(tmp, 0)
    else:
        n_o = [ max_points ]
        n_o = torch.cuda.IntTensor(n_o)
        coord = input[0]
    feat = coord
    label =  n_o
    return coord, feat, label # (n, 3), (n, c), (b)

def pt_fmap_trans(input, max_points):
    p_all, x = input.size(2),input.size(1)
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.transpose(0,1)
    return f

def pt_fmap_trans_ot(input, max_points):
    p_all, x = input.size(2),input.size(1)
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.permute(1,2,0)
    return f


class GMPTV(nn.Module):
    def __init__(self,
                 backbone=None,
                 feature_channels=128,
                 ffn_dim_expansion=4,
                 num_transformer_pt_layers=1,
                 num_transformer_layers=8,
                 ):
        super(GMPTV, self).__init__()

        self.backbone = backbone
        self.feature_channels = feature_channels

        # PointNet
        if self.backbone=='pointnet':
            self.pointnet = PointNet(output_channels = self.feature_channels)
            feature_channels = self.feature_channels
        #MLP
        if self.backbone=='mlp':
            self.mlp = MLP(output_channels = self.feature_channels)
            feature_channels = self.feature_channels
        # DGCNN
        if self.backbone=='DGCNN':
            self.DGCNN = DGCNN(output_channels = self.feature_channels)
            feature_channels = self.feature_channels

        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_pt_layers = num_transformer_pt_layers

        # Transformer
        if self.num_transformer_layers > 0:
            self.transformer = FeatureTransformer3D(num_layers=num_transformer_layers,
                                                d_model=feature_channels,
                                                ffn_dim_expansion=ffn_dim_expansion,
                                                )

        # self correlation with self-feature similarity
        self.feature_flow_attn = SelfCorrelationSoftmax3D(in_channels=feature_channels)

    def forward(self, pc0, pc1,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        pc0 = torch.permute(pc0, (0, 2, 1)) # torch.Size([batch, n_point, 3]) -> torch.Size([batch, 3, n_point])
        pc1 = torch.permute(pc1, (0, 2, 1)) # torch.Size([batch, n_point, 3]) -> torch.Size([batch, 3, n_point])

        xyzs1, xyzs2 = pc0, pc1
        if self.backbone=='pointnet':
            feats0_3d = self.pointnet(xyzs1)
            feats1_3d = self.pointnet(xyzs2)
        if self.backbone=='mlp':
            feats0_3d = self.mlp(xyzs1)
            feats1_3d = self.mlp(xyzs2)
        if self.backbone=='DGCNN':
            feats0_3d = self.DGCNN(xyzs1)
            feats1_3d = self.DGCNN(xyzs2)

        flow = None

        feature0, feature1 = feats0_3d, feats1_3d
        # Transformer
        if self.num_transformer_pt_layers > 0:
            feature0, feature1 = self.transformer_PT(xyzs1, xyzs2,
                                                    feature0, feature1,
                                                    )
        if self.num_transformer_layers > 0:
            feature0, feature1 = self.transformer(feature0, feature1,
                                                )

        # global matching
        # flow prediction with cros-attn
        flow_pred = global_correlation_softmax_3d(feature0, feature1, xyzs1, xyzs2)[0]
        flow = flow + flow_pred if flow is not None else flow_pred
        flow = self.feature_flow_attn(feature0, flow)
        flow_preds.append(flow)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict
