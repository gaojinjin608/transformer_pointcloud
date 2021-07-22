import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
from util import sample_and_group_mutilscale

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()

        # multi-scale
        self.conv11 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv12 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.conv21 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv22 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.conv31 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv32 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn11 = nn.BatchNorm1d(out_channels)
        self.bn12 = nn.BatchNorm1d(out_channels)

        self.bn21 = nn.BatchNorm1d(out_channels)
        self.bn22 = nn.BatchNorm1d(out_channels)

        self.bn31 = nn.BatchNorm1d(out_channels)
        self.bn32 = nn.BatchNorm1d(out_channels)

        self.conv = nn.Conv1d(3 * out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, x2, x3):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 

        #scale 1
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn11(self.conv11(x))) # B, D, N
        x = F.relu(self.bn12(self.conv12(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        #scale 2
        x2 = x2.permute(0, 1, 3, 2)   
        x2 = x2.reshape(-1, d, s) 
        x2 = F.relu(self.bn21(self.conv11(x2))) # B, D, N
        x2 = F.relu(self.bn22(self.conv12(x2))) # B, D, N
        x2 = F.adaptive_max_pool1d(x2, 1).view(batch_size, -1)
        x2 = x2.reshape(b, n, -1).permute(0, 2, 1)
        #scale 3
        x3 = x3.permute(0, 1, 3, 2)   
        x3 = x3.reshape(-1, d, s) 
        x3 = F.relu(self.bn31(self.conv11(x3))) # B, D, N
        x3 = F.relu(self.bn32(self.conv12(x3))) # B, D, N
        x3 = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1)
        x3 = x3.reshape(b, n, -1).permute(0, 2, 1)

        x = torch.cat([x, x2, x3], -1) # B, N, 3D
        x = F.relu(self.bn(self.conv(x.permute(0, 2, 1))))
        x = x.permute(0, 2, 1)  # B, N, D

        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature, new_feature2, new_feature3 = sample_and_group_mutilscale(npoint=512, radius=0.15, nsample=[16, 32, 64], xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature, new_feature2, new_feature3)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature, new_feature2, new_feature3 = sample_and_group_mutilscale(npoint=256, radius=0.2, nsample=[16, 32, 64], xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature, new_feature2, new_feature3)

        x = self.pt_last(feature_1, new_xyz)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        xyz = xyz.permute(0, 2, 1)
        xyz = self.pos_xyz(xyz)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        # 4 heads
        b, c, n = x.size() # b, c, n
        x = x + xyz  # positional encoding
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_q = torch.reshape(x_q, [b, 4, n, c // 4])  # b, 4, n, c/4

        x_k = self.k_conv(x)  # b, c, n
        x_k = torch.reshape(x_k, [b, 4, c // 4, n])  # b, 4, c/4, n
        x_v = self.v_conv(x)
        x_v = torch.reshape(x_v, [b, 4, c // 4, n])  # b, 4, c/4, n
        # b, n, n
        energy = torch.matmul(x_q, x_k)  # b, 4, n, n

        attention = self.softmax(energy, -1)
        attention = self.softmax(attention, -2)  # b, 4, n, n
        # b, c, n
        x_r = torch.matmul(x_v, attention).reshape(b, c, n)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r # residual
        return x
