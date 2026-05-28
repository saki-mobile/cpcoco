import logging
import torch
import statistics
import torch.nn as nn
import torch.nn.functional as F
from opencood.utils.visual_utils import plot_point
from opencood.utils.matcher_utils import MST

# size_x, size_y = 96, 352
size_x, size_y = 80, 80

def get_valid_dict(match_dict, pos_radius):
    valid_match_dict = {}
    inner_similarity = []

    for p_s, p_t in zip(match_dict['point0'], match_dict['point1_transed']):
        dists = torch.cdist(p_s, p_t, p=2)
        mask = (dists < pos_radius) * \
                (dists == torch.min(dists, dim=-1, keepdim=True)[0]) * \
                (dists == torch.min(dists, dim=0, keepdim=True)[0])
        simi = (2*torch.nonzero(mask).size(0))/(mask.size(0) + mask.size(1))
        inner_similarity.append(simi)

    _, mst_pair_idx, _ = MST(match_dict.pop('pair').cpu().numpy(), inner_similarity, threshold=1e-5, type='simi')

    if len(mst_pair_idx) > 0:
        for key in match_dict.keys():
            valid_match_dict.update({key: [match_dict[key][i] for i in mst_pair_idx]})

    return valid_match_dict
    

class MatchLoss(nn.Module):
    def __init__(self, args):
        super(MatchLoss, self).__init__()
        self.args = args
        self.infonce_loss = InfoNCELossFull(r_p=args['r_p'], r_n=args['r_n'])

    def forward(self, loss_dict):
        match_loss_dict = {}
        
        if loss_dict.get('match_loss', None) is not None:
            match_dict = loss_dict['match_loss']
            assert len(match_dict['point0']) == len(match_dict['point1']), 'not equal'

            valid_match_dict = get_valid_dict(match_dict, self.args['r_p'])

            if len(valid_match_dict) == 0:
                point_circle_loss = torch.tensor(0.).cuda()
                point_overlap_loss = torch.tensor(0.).cuda()
                match_loss_dict.update({'circle_loss': point_circle_loss.item(),
                                        'overlap_loss': point_overlap_loss.item()})
            else:
                point_circle_loss = self.infonce_loss(
                                        valid_match_dict['point0'],
                                        valid_match_dict['point1_transed'],
                                        valid_match_dict['match_logits']) * 0.1
                point_overlap_loss = torch.tensor(0.).cuda()
                # visual_matching(valid_match_dict['point0'],
                #                 valid_match_dict['point1'],
                #                 valid_match_dict['point1_transed'],
                #                 valid_match_dict['corr_index'],
                #                 valid_match_dict['valid_corr'])
                match_loss_dict.update({'circle_loss': point_circle_loss.item(),
                                        'overlap_loss': point_overlap_loss.item()})
                
        return match_loss_dict, point_circle_loss, point_overlap_loss

    
class InfoNCELossFull(nn.Module):
    """Computes InfoNCE loss
    """
    def __init__(self, r_p, r_n):
        """
        Args:
            d_embed: Embedding dimension
            r_p: Positive radius (points nearer than r_p are matches)
            r_n: Negative radius (points nearer than r_p are not matches)
        """

        super().__init__()
        self.r_p = r_p
        self.r_n = r_n
        self.tempreture = 0.5

    def compute_infonce(self, anchor_xyz, positive_xyz, match_logits):
        """

        Args:
            anchor_feat: Shape ([B,] N_anc, D)
            positive_feat: Shape ([B,] N_pos, D)
            anchor_xyz: ([B,] N_anc, 3)
            positive_xyz: ([B,] N_pos, 3)

        Returns:
        """
        with torch.no_grad():
            dist_keypts = torch.cdist(anchor_xyz, positive_xyz)

            dist1, idx1 = dist_keypts.topk(k=1, dim=-1, largest=False)  # Finds the positive (closest match)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences (..., N_anc)
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary,
            ignore.scatter_(-1, idx1, 0)     # except the positive (..., N_anc, N_pos)

        match_logits /= self.tempreture
        match_logits[..., ignore] = -float('inf')

        loss = -torch.gather(match_logits, -1, idx1).squeeze(-1) + torch.logsumexp(match_logits, dim=-1)
        loss = torch.sum(loss[mask]) / torch.sum(mask)

        return loss

    def forward(self, src_xyz, tgt_xyz, match_logits):
        """
        Args:
            src_feat: List(B) of source features (N_src, D)
            tgt_feat: List(B) of target features (N_tgt, D)
            src_xyz:  List(B) of source coordinates (N_src, 3)
            tgt_xyz: List(B) of target coordinates (N_tgt, 3)

        Returns:

        """
        B = len(src_xyz)
        infonce_loss = [self.compute_infonce(src_xyz[b], tgt_xyz[b], match_logits[b]) for b in range(B)]

        return torch.mean(torch.stack(infonce_loss))
    

def visual_matching(point0, point1, point1_transed, corr_index, valid_corr):
    for b in range(len(point0)):
        # gt_dist = torch.cdist(point0[b], point1_transed[b])
        # gt_mask = (gt_dist < 0.8) * \
        #     (gt_dist == torch.min(gt_dist, dim=-1, keepdim=True)[0]) * \
        #     (gt_dist == torch.min(gt_dist, dim=0, keepdim=True)[0])

        # plot_point(point0[b], size_x, size_y, point1_transed[b], gt_mask, prefix=f'gt_{b}')

        # valid_index = corr_index[b][valid_corr[b]]
        # valid_mask = torch.zeros((point0[b].size(0), point1[b].size(0)))
        # valid_mask[valid_index[:, 0], valid_index[:, 1]] = 1.
        # plot_point(point0[b], size_x, size_y, point1_transed[b], valid_mask, prefix=f'pred_{b}')

        # valid_index = corr_index[b]
        # valid_mask = torch.zeros((point0[b].size(0), point1[b].size(0)))
        # valid_mask[valid_index[:, 0], valid_index[:, 1]] = 1.
        # plot_point(point0[b], size_x, size_y, point1_transed[b], valid_mask, prefix=f'pred_r_{b}')

        pts0, pts1, pts1_trans = point0[b], point1[b], point1_transed[b]
        valid_index = corr_index[b][valid_corr[b]]
        valid_dist = torch.norm(pts0[valid_index[:, 0]]-pts1_trans[valid_index[:, 1]], dim=1) < 1.6
        from opencood.utils.visual_utils import plot_correspondence
        plot_correspondence(pts0.detach().cpu().numpy(), pts1.detach().cpu().numpy(), torch.cat((valid_index, valid_dist.unsqueeze(1).int()), dim=1))
       