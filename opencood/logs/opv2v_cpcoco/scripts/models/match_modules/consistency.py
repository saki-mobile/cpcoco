import torch
from torch import nn

class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention):
        """
        Input:
            - feat:     [1, num_channels, num_corr]  input feature
            - attention [1, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [1, num_channels, num_corr]  updated feature
        """
        num_corr = feat.shape[-1]
        Q = self.projection_q(feat).view([self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('hco, hci->hoi', Q, K) / (self.num_channels // self.head) ** 0.5
        # combine the feature similarity with spatial consistency
        weight = torch.softmax(attention * feat_attention, dim=-1)
        message = torch.einsum('hoi, hci-> hco', weight, V).reshape([1, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res 


class NonLocalNet(nn.Module):
    def __init__(self, num_layers=6, num_channels=128):
        super(NonLocalNet, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleDict()
        self.layer0 = nn.Conv1d(4, num_channels, kernel_size=1, bias=True)
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True)
            )
            self.blocks[f'PointCN_layer_{i}'] = layer
            self.blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(num_channels)

        
        self.classification = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True),
        )

    def forward(self, feat, corr_compatibility):
        """
        Input: 
            - corr_feat:          [1, in_dim, num_corr]   input feature map
            - corr_compatibility: [1, num_corr, num_corr] spatial consistency matrix 
        Output:
            - feat:               [num_channels, num_corr] updated feature
        """
        feat = self.layer0(feat.t().unsqueeze(0))
        corr_compatibility = corr_compatibility.unsqueeze(0)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)

        confidence = self.classification(feat)
        return feat.squeeze(0).t(), confidence.squeeze(0).t() 