import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.match_modules.loftr_resnet import LoftrResnet, OverlapHead

from opencood.utils.matcher_utils import get_voxel_coordinate, get_fore_point


class CPSCoCoMatcher(nn.Module):
    def __init__(self, matcher_args) -> None:
        super(CPSCoCoMatcher, self).__init__()
        self.mather_args = matcher_args
        self.backbone = ResNetBEVBackbone(matcher_args['backbone'], 64)
        self.resnet = LoftrResnet(matcher_args['resnetfpn'])
        self.overlap_head = OverlapHead(64)

    def forward(self, feature_2d, foreground_index, affine_matrix, pair, **kwargs):
        """
            通过特征找到前景Keypoints, 通过sp和lg完成。
            Args:
                feature map single (N, C, 96, 352): voxelization feature.
                feature index (K, 4): feature froeground index.
                paired transformation matrix (1, L, L, 2, 3): gt generator.
                pair (P, 2): collab pairs.
            Returns:
                match (P, m, n): match result.
                ktps_2d_src (P, 2): src_kpts.
                kpts_2d_tgt (P, 2): tgt_kpts.
                loss:
                    loss_dict = dict{'point cls loss': {'lable': 预测keypoins, 'gt': 取到keypoints的真实标签},
                        'circle loss': {'keypoint feat': cross-attn(keypoint feat), 'keypoint pos': coord in same coords system},
                        'dist loss': dist loss,
                        'matching loss': nll loss}
        """
        loss_dict = {}
        device = feature_2d.device

        '''
            这里整个部分的逻辑有点问题，先做一个3*3的卷积之后，再加了一个7*7的卷积。在后面加这种东西，会让形状变得不太正常。
            所以这样，首先在下采样1/2的情况下，拿到前景点做cross，然后再按照Loftr做多层次特征提取，这样会好一些。
        '''
        feature_2d = self.backbone({'spatial_features': feature_2d})['spatial_features_2d']
        N, _, H, W = feature_2d.shape
        foreground_mask = torch.zeros((N, 1, H*2, W*2), device=device)
        foreground_mask[foreground_index[:, 0], 0, foreground_index[:, -2], foreground_index[:, -1]] = 1.
        foreground_mask = F.max_pool2d(foreground_mask, 2, 2)

        _, feature_2d = self.resnet(feature_2d)

        fg_mask0 = foreground_mask.index_select(0, pair[0])
        fg_mask1 = foreground_mask.index_select(0, pair[1])

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        position = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(pair.size(1), 1, 1, 1).double().cuda()

        grid0 = get_voxel_coordinate(position.detach().clone())
        grid1 = get_voxel_coordinate(position.detach().clone())
        grid1_transed = get_voxel_coordinate(position.detach().clone(), affine_matrix[0, pair[1], pair[0]])

        point0, point1 = get_fore_point(grid0, fg_mask0), get_fore_point(grid1, fg_mask1)
        point1_transed = get_fore_point(grid1_transed, fg_mask1)

        inference_data = None
        feat0 = feature_2d.index_select(0, pair[0])
        feat1 = feature_2d.index_select(0, pair[1])        

        # Full Convolution Model
        match_logits_list, corr_index_list, valid_corr_list = [], [], []
        for b, (feat_pair, mask_pair, grid_pair) in enumerate(zip(torch.cat((feat0.unsqueeze(1), feat1.unsqueeze(1)), dim=1),
                                                                torch.cat((fg_mask0.unsqueeze(1), fg_mask1.unsqueeze(1)), dim=1),
                                                                torch.cat((grid0.unsqueeze(1), grid1.unsqueeze(1)), dim=1))):
            # _, feat_pair = self.resnet(feat_pair)
            
            match_logits, corr_index, valid_corr = \
                    self.overlap_head(feat_pair, mask_pair, grid_pair)

            match_logits_list.append(match_logits)
            corr_index_list.append(corr_index)
            valid_corr_list.append(valid_corr)


        loss_dict.update({
            'match_loss':{
                'match_logits': match_logits_list,
                'corr_index': corr_index_list,
                'valid_corr': valid_corr_list,
                'point0': point0,
                'point1': point1,
                'point1_transed': point1_transed,
                'pair': pair
            }
        })   

        if not kwargs.get('is_train', True):
            if len(point0) == len(point1) != 0:  
                inference_data = {
                    'corr_index': corr_index_list,
                    'valid_corr': valid_corr_list,
                    'point0': point0,   
                    'point1': point1,
                    'point1_transed': point1_transed,
                    'H': H,
                    'W': W
                }

        return inference_data, loss_dict