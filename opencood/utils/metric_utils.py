import torch
import statistics
import numpy as np
from scipy.linalg import logm
from opencood.utils.matcher_utils import MST
import cv2
import time


def get_valid_dict(match_dict):
    valid_match_dict = {}
    inner_similarity = []

    for p_s, p_t, valid_corr in zip(match_dict['point0'], match_dict['point1'], match_dict['valid_corr']):
        simi = sum(valid_corr).item() / (len(p_s) + len(p_t))
        inner_similarity.append(simi)

    pair = match_dict.pop('pair').cpu().numpy()
    _, mst_pair_idx, _ = MST(pair, inner_similarity, type='simi', threshold=0)

    if len(mst_pair_idx) > 0:
        for key in match_dict.keys():
            valid_match_dict.update({key: [match_dict[key][i] for i in mst_pair_idx]})
    valid_match_dict.update({'pair': pair[:, mst_pair_idx]})

    return valid_match_dict

        
def normalize_pred_matrix(pred_matrix, pair, H, W):
    n = pred_matrix.shape[0]
    matrix_n_4x4 = torch.eye(4).unsqueeze(0).repeat(n, 1, 1)  # 创建 n 个 4x4 的单位矩阵

    matrix_n_4x4[:, :2, :2] = pred_matrix[:, :2, :2]
    matrix_n_4x4[:, :2, -1] = pred_matrix[:, :2, -1]

    pairwise_t_matrix = torch.eye(4).repeat(5, 5, 1, 1).unsqueeze(0)
    pairwise_t_matrix[0, pair[0], pair[1]] = matrix_n_4x4

    pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
    pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
    pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / W
    pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / H

    return pairwise_t_matrix


def get_metrics(metric_dict):
    H, W = metric_dict.pop('H'), metric_dict.pop('W')
    voxel_size = metric_dict.pop('voxel')
    valid_pair = np.empty((2, 0))
    correct_sum, failed_sum = 0, 0
    rtes, rres, t_matrix_pred_list = [], [], []
    
    # lidar_np = metric_dict.pop('lidar_np')

    metric_dict = get_valid_dict(metric_dict)
    if len(metric_dict) == 1:
        return torch.eye(4)[[0, 1]][:, [0, 1, 3]].repeat(5, 5, 1, 1).unsqueeze(0).cuda(), valid_pair, {'rre': rres, 'rte': rtes, 'correct': correct_sum, 'failed': failed_sum, 'time_post': []}

    time_post = []
    for pts0, pts1, pts1_transed, corr_index, valid_corr, t_matrix, pair in zip(metric_dict['point0'], metric_dict['point1'], metric_dict['point1_transed'], 
                                                    metric_dict['corr_index'], metric_dict['valid_corr'], metric_dict['trans_matrix'], metric_dict['pair'].T):
        start = time.time()
        
        valid_index = corr_index[valid_corr]
        
        # valid_index = corr_index

        src_p = pts0[valid_index[:, 0]]
        tgt_p = pts1[valid_index[:, 1]]
        tgt_transed = pts1_transed[valid_index[:, 1]]
        src_p[:, 0] = ((src_p[:, 0] / (W - 1)) * 2 - 1) * (W * voxel_size)
        src_p[:, 1] = ((src_p[:, 1] / (H - 1)) * 2 - 1) * (H * voxel_size)
        tgt_p[:, 0] = ((tgt_p[:, 0] / (W - 1)) * 2 - 1) * (W * voxel_size)
        tgt_p[:, 1] = ((tgt_p[:, 1] / (H - 1)) * 2 - 1) * (H * voxel_size)
        tgt_transed[:, 0] = ((tgt_transed[:, 0] / (W - 1)) * 2 - 1) * (W * voxel_size)
        tgt_transed[:, 1] = ((tgt_transed[:, 1] / (H - 1)) * 2 - 1) * (H * voxel_size)

        if len(src_p) == len(tgt_p) <= 10:
            valid_pair = np.concatenate((valid_pair, np.empty((2, 0))), axis=1)
            time_post.append((time.time() - start) * 1000)
            continue
        else:
  
            t_matrix_pred, _ = cv2.estimateAffinePartial2D(src_p.detach().cpu().numpy(), 
                                                        tgt_p.detach().cpu().numpy())
     
            # t_matrix_pred, _ = cv2.estimateAffinePartial2D(src_p.detach().cpu().numpy(), 
            #                                             tgt_p.detach().cpu().numpy(), method=cv2.RANSAC,    
            #                                             ransacReprojThreshold=1,
            #                                             maxIters=10000,
            #                                             confidence=0.99)
            time_post.append((time.time() - start) * 1000)
            
            rre, rte, success = calculate_rre_rte_success_rate(t_matrix, t_matrix_pred, 3, 3)
            rtes.append(rte)
            rres.append(rre)
            t_matrix_pred_list.append(torch.from_numpy(t_matrix_pred).float())
            valid_pair = np.concatenate((valid_pair, pair.reshape(-1, 1)), axis=1)

            if success: 
                correct_sum += 1
            else:
                failed_sum += 1 

    if len(t_matrix_pred_list) > 0:
        print(f"rte:{statistics.mean(rtes):.5f}, rre:{statistics.mean(rres):.5f}\ncorrect:{correct_sum}, failed:{failed_sum}")
        pred_affine_matrix = normalize_pred_matrix(torch.stack(t_matrix_pred_list), valid_pair, H*voxel_size, W*voxel_size).cuda()
        return pred_affine_matrix, valid_pair, {'rre': rres, 'rte': rtes, 'correct': correct_sum, 'failed': failed_sum, 'time_post': time_post}
    else:
        return torch.eye(4)[[0, 1]][:, [0, 1, 3]].repeat(5, 5, 1, 1).unsqueeze(0).cuda(), valid_pair, {'rre': rres, 'rte': rtes, 'correct': correct_sum, 'failed': failed_sum, 'time_post': time_post}
    

def calculate_rre_rte_success_rate(matrix_gt, matrix_est, rotation_threshold_deg=3, translation_threshold=3):
    matrix_est = matrix_est
    matrix_gt = matrix_gt.cpu().numpy()
    
    # Extract the rotation and translation parts of the matrices, ignoring z-axis
    R_gt = matrix_gt[:2, :2]
    R_est = matrix_est[:2, :2]

    t_gt = matrix_gt[:2, 3]
    t_est = matrix_est[:2, 2]

    # Calculate R1^T * R2
    relative_rotation = np.transpose(R_est) @ R_gt

    # Calculate matrix logarithm
    log_rotation = logm(relative_rotation, disp=False)[0]

    # Calculate Frobenius norm
    rre = np.linalg.norm(log_rotation, 'fro')
    rre_degress = np.degrees(rre)

    # Calculate Relative Translation Error (RTE)
    rte = np.linalg.norm(t_gt - t_est)

    # Calculate Success Rate
    rre_success = rre_degress <= rotation_threshold_deg
    rte_success = rte <= translation_threshold
    success = rre_success and rte_success
    

    return rre_degress, rte, success