import torch.nn as nn
import torch
import torch.nn.functional as F
from opencood.models.match_modules.attention import GCN
from opencood.utils.matcher_utils import get_fore_point


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(planes)
        self.batch_norm2 = nn.BatchNorm2d(planes)

        self.down = conv1x1(in_planes, planes, stride=stride)
        self.down_norm = nn.BatchNorm2d(planes)

    def forward(self, x):    
        y = x

        y = self.conv1(y)
        y = self.batch_norm1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.batch_norm2(y)

        if self.stride != 1:
            x = self.down(x)
            x = self.down_norm(x)
        
        return self.relu(x + y)
    

class LoftrResnet(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(64, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4

        # FPN upsample
        self.layer2_conv = conv1x1(block_dims[1], block_dims[1])
        self.layer2_up = nn.ConvTranspose2d(block_dims[1], block_dims[1], kernel_size=2, stride=2)
        self.layer2_outconv = conv1x1(block_dims[0], block_dims[1]) 
        self.layer2_conv2_0 = conv3x3(block_dims[1], block_dims[1])
        self.batch_norm2 = nn.BatchNorm2d(block_dims[1])
        self.layer2_relu = nn.LeakyReLU()
        self.layer2_conv2_1 = conv3x3(block_dims[1], block_dims[0])

        self.layer1_up = nn.ConvTranspose2d(block_dims[0], block_dims[0], kernel_size=2, stride=2)
        self.layer1_conv = conv1x1(block_dims[0], block_dims[0])
        self.layer1_conv1 = conv3x3(block_dims[0], block_dims[0])
        self.batch_norm3 = nn.BatchNorm2d(block_dims[0])
        self.layer1_relu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x_row = x

        # ResNet Backbone
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x0 = self.relu(x)

        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4

        x2_out = self.layer2_conv(x2)
        # FPN
        # x2_out_1x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out_1x = self.layer2_up(x2_out)
        x1_out = self.layer2_outconv(x1)
        x1_out = self.layer2_conv2_0(x1_out + x2_out_1x)
        x1_out = self.batch_norm2(x1_out)
        x1_out = self.layer2_relu(x1_out)

        x1_out = self.layer2_conv2_1(x1_out)

        # x1_out_x = F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out_x = self.layer1_up(x1_out)
        x_row = self.layer1_conv(x_row)
        x_out = self.layer1_conv1(x_row + x1_out_x)
        x_out = self.batch_norm3(x_out)
        x_out = self.relu(x_out)

        return x2_out, x_out


class OverlapHead(nn.Module):
    def __init__(self, middle_dim):
        super(OverlapHead, self).__init__()
        self.gnn = GCN(4, middle_dim, 2)
        self.proj_gnn = nn.Conv1d(middle_dim, middle_dim, kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(middle_dim, 1, kernel_size=1, bias=True)
        self.overlap_score = nn.Linear(middle_dim+2, middle_dim, bias=False)

        self.epsilon = nn.Parameter(torch.tensor(-5.0))
        self.W = torch.nn.Parameter(torch.zeros(middle_dim, middle_dim), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.W, std=0.1)

    def forward(self, feat_c, mask, grid):
        """
        Args:
            feat_c: Input features [2, C, H, W]
            mask: Binary mask indicating valid regions [2, 1, H, W]
        """
        p0, p1 = get_fore_point(grid[:1].repeat(2, 1, 1, 1), mask)
        src_feats, tgt_feats = get_fore_point(feat_c, mask)
        len_src_c = src_feats.size(0)

        src_feats, tgt_feats = self.gnn(p0, p1, src_feats, tgt_feats)
        feats_c = self.proj_gnn(torch.cat((src_feats, tgt_feats), dim=0).t().unsqueeze(0))

        scores_c = self.proj_score(feats_c)

        feats_gnn_norm = F.normalize(feats_c.squeeze(0).t(), p=2, dim=-1)
        feats_gnn_raw = feats_c.squeeze(0).t()
        scores_c_raw = scores_c.squeeze(0).t()
        
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.t())
        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]
        
        temperature = torch.exp(self.epsilon) + 0.03
        s1 = torch.matmul(F.softmax(inner_products / temperature ,dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0,1) / temperature,dim=1), src_scores_c)
        scores_saliency = torch.cat((s1,s2),dim=0)

        x = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)
        x = self.overlap_score(x)

        feats = x[:, ]

        anchor_feat, positive_feat = feats[:len_src_c], feats[len_src_c:]

        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        match_logits = torch.einsum('...ic,cd,...jd->...ij', anchor_feat, W_symmetrical, positive_feat)  # (..., N_anc, N_pos)

        with torch.no_grad():
            topk_value, topk_index = torch.topk(match_logits.clone(), 10, dim=1)
            match_logits_topk = torch.zeros_like(match_logits)
            match_logits_topk.scatter_(1, topk_index, topk_value)

            corr_mask = (match_logits_topk > 0) \
                * (match_logits_topk == torch.max(match_logits_topk, dim=-1, keepdim=True)[0]) \
                * (match_logits_topk == torch.max(match_logits_topk, dim=0, keepdim=True)[0])
            
            corr_index = corr_mask.nonzero()
            src_kpts, tgt_kpts = p0.clone()[corr_index[:, 0], :].float(), p1.clone()[corr_index[:, 1], :].float()

            src_dist = torch.norm((src_kpts[:, None, :] - src_kpts[None, :, :]), dim=-1)
            corr_compatibility = src_dist - torch.norm((tgt_kpts[:, None, :] - tgt_kpts[None, :, :]), dim=-1)

            corr_compatibility = corr_compatibility ** 2

            match_valid = torch.sum(corr_compatibility <= 0.05, dim=1)
            valid_corr = match_valid > (len(corr_index) * 0.05)

        return match_logits, corr_index, valid_corr
    
