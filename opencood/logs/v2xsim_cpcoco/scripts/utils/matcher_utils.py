import torch
import numpy as np
import networkx as nx
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def MST(pairs, similarity_scores, threshold=0.08, type='mst'):
    similarity_scores = np.array(similarity_scores)
    assert pairs.shape[0] == 2, "Pairs must have shape (2, N)"
    assert pairs.shape[1] == similarity_scores.shape[0], "Length of similarity_scores must match number of pairs"

    if type == 'mst':
        G = nx.Graph()
        for i in range(pairs.shape[1]):
            node1 = pairs[0, i]
            node2 = pairs[1, i]
            score = similarity_scores[i]

            if score >= threshold:
                G.add_edge(node1, node2, weight=-score, index=i)  # 使用负分数以最小化权重
        
        mst = nx.minimum_spanning_tree(G)
        
        mst_edges = []
        original_indices = []
        
        for edge in mst.edges(data=True):
            node1, node2, data = edge
            mst_edges.append((node1, node2))
            original_indices.append(data['index'])
        
        # 将边和索引转换为 numpy 数组，并转置以符合 (2, M) 的形状
        mst_edges = np.array(mst_edges).T  # 边的形状为 (2, M)
        original_indices = np.array(original_indices)  # 索引的一维数组

        if len(mst_edges) != 0:
            mst_score = similarity_scores[original_indices]
        else:
            mst_score = []
    elif type == 'simi':
        keep_pair = similarity_scores > threshold
        mst_edges = pairs[:, keep_pair]
        original_indices = keep_pair.nonzero()[0]
        mst_score = similarity_scores[keep_pair]
    else: 
        raise TypeError()
    return mst_edges, original_indices, mst_score



def noise_transformation(loc_noise=10, yaw_noise=5, H=32, W=32):
    # gps is a list in the format of [x, y, z, pitch, roll, yaw]
    noise = np.zeros((6))
    noise[0] += np.random.normal(0, loc_noise)
    noise[1] += np.random.normal(0, loc_noise)
    noise[5] += np.random.normal(0, yaw_noise)

    gps = noise

    # used for rotation matrix
    c_y = np.cos(np.radians(gps[5]))
    s_y = np.sin(np.radians(gps[5]))
    c_r = np.cos(np.radians(gps[4]))
    s_r = np.sin(np.radians(gps[4]))
    c_p = np.cos(np.radians(gps[3]))
    s_p = np.sin(np.radians(gps[3]))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = gps[0]
    matrix[1, 3] = gps[1]
    matrix[2, 3] = gps[2]

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r


    pairwise_t_matrix = matrix[[0, 1],:][:,[0, 1, 3]] # [B, L, L, 2, 3]
    pairwise_t_matrix[0,1] = pairwise_t_matrix[0,1] * H / W
    pairwise_t_matrix[1,0] = pairwise_t_matrix[1,0] * W / H
    pairwise_t_matrix[0,2] = pairwise_t_matrix[0,2] / W
    pairwise_t_matrix[1,2] = pairwise_t_matrix[1,2] / H

    normalized_affine_matrix = torch.from_numpy(pairwise_t_matrix).cuda()


    return normalized_affine_matrix


def get_fore_point(grids, foregrounds, pixel=False):
    fore_point = []
    for grid, foreground in zip(grids, foregrounds):
        index = torch.nonzero(foreground)
        info = grid[:, index[:, -2], index[:, -1]]
        if pixel:
            info[0] = torch.round(info[0])
            info[1] = torch.round(info[1])
        fore_point.append(info.t())
    return fore_point


def get_voxel_coordinate(grids, affine_matrix=None):
    N, _, H, W = grids.shape

    grids[:, 0, ] = (grids[:, 0, ] / (W - 1)) * 2 - 1
    grids[:, 1, ] = (grids[:, 1, ] / (H - 1)) * 2 - 1
    if affine_matrix is not None:
        grids = torch.cat([grids, 
                        torch.ones((N, 1, H, W)).cuda()], dim=1)
        grids = torch.einsum('nij,njhw->nihw', affine_matrix, grids)

    grids[:, 0, ] = ((grids[:, 0, ] + 1) / 2) * (W - 1)
    grids[:, 1, ] = ((grids[:, 1, ] + 1) / 2) * (H - 1)
    return grids


def compute_match_matrix(points, ref_points):
    """
    根据浮点型点的坐标生成周围的整数点，并与参考点计算匹配矩阵。
    Args:
        points (torch.Tensor): 输入点坐标 (n*2)。
        ref_points (torch.Tensor): 参考点坐标 (m*2)。
    Returns:
        torch.Tensor: 匹配矩阵，形状为 (n*m)。
    """
    # 获取 n*4*2 的周围整数点坐标
    floor_points = torch.floor(points).long()  # 向下取整
    ceil_points = floor_points + 1  # 向上取整

    x1, y1 = floor_points[:, 0], floor_points[:, 1]
    x2, y2 = ceil_points[:, 0], ceil_points[:, 1]

    integer_coordinates = torch.stack([
        torch.stack([x1, y1], dim=-1),  # 左下角
        torch.stack([x1, y2], dim=-1),  # 左上角
        torch.stack([x2, y1], dim=-1),  # 右下角
        torch.stack([x2, y2], dim=-1)  # 右上角
    ], dim=1)  # n*4*2

    # 初始化 n*m 的匹配矩阵
    n, m = points.size(0), ref_points.size(0)
    match_matrix = torch.zeros((n, m), dtype=torch.int, device=points.device)

    # 逐个整数点计算与 ref_points 的距离
    for i in range(4):  # 遍历 n*4*2 的每个整数点
        int_coords = integer_coordinates[:, i, :]  # n*2
        distances = torch.cdist(int_coords.float(), ref_points.float())  # n*m
        match_matrix |= (distances == 0).int()  # 若距离为 0，对应位置设置为 1

    return match_matrix


def cdist(a, b, metric='euclidean'):
    if metric == 'cosine':
        return torch.sqrt(2 - 2 * torch.matmul(a, b.transpose(2, 1)))
    elif metric == 'arccosine':
        return torch.acos(torch.matmul(a, b.transpose(2, 1)))
    else:
        # diffs = torch.unsqueeze(a, dim=2) - torch.unsqueeze(b, dim=1)
        diffs = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)
        if metric == 'sqeuclidean':
            return torch.sum(diffs**2, dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-12)
        elif metric == 'cityblock':
            return torch.sum(torch.abs(diffs), dim=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.
                format(metric))
        

def circle_full_cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.
    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.
    Args:
        a: The left-hand side, shaped ([*,] F, B1).  <- Not that dimension ordering is different from torch.cdist
        b: The right-hand side, shaped ([*,], F, B2).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.

    Taken from Predator source code, which was modified from D3Feat.
    """
    if metric == 'sqeuclidean':
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sum(diffs ** 2, dim=-3)
    elif metric == 'euclidean':
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sqrt(torch.sum(diffs ** 2, dim=-3) + 1e-12)
    elif metric == 'cityblock':
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sum(torch.abs(diffs), dim=-3)
    elif metric == 'cosine':
        numer = a.transpose(-1, -2) @ b
        denom = torch.clamp_min(
            torch.norm(a, dim=-2)[..., :, None] * torch.norm(b, dim=-2)[..., None, :],
            1e-8)
        dist = 1 - numer / denom
        return dist
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))