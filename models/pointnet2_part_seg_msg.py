"""
pointnet2_part_seg_msg.py —— PointNet++ 部件分割模型（MSG 多尺度分组）
================================================================
网络结构: 编码-解码（U-Net 风格）
  编码器（SA，下采样）:
    SA1 (MSG): 2048→512点，3个半径，输出 320 维
    SA2 (MSG): 512→128点，2个半径，输出 512 维
    SA3 (全局): 128→1点，输出 1024 维全局特征
  解码器（FP，上采样 + 跳跃连接）:
    FP3: l3→l2，插值 + 拼接 l2 特征 → 256 维
    FP2: l2→l1，插值 + 拼接 l1 特征 → 128 维
    FP1: l1→l0，插值 + 拼接 [类别one-hot + 坐标 + l0特征] → 128 维
  输出: Conv1d(128→num_classes)，每个点一个类别分数
  num_classes=50（ShapeNetPart 共 50 个部件类别）
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    """ PointNet++ 部件分割模型主体 """

    def __init__(self, num_classes, normal_channel=False):
        """
        参数:
            num_classes: 部件类别数（ShapeNetPart 为 50）
            normal_channel: 是否使用法向量
        """
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3  # 法向 3 维
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        # ===== 编码器（集合抽象 SA，逐步下采样） =====
        # SA1: 512点，3个半径(0.1/0.2/0.4)，输出 64+128+128=320 维
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # SA2: 128点，2个半径(0.4/0.8)，输出 256+256=512 维
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # SA3: 全局分组，输出 1024 维
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        # ===== 解码器（特征传播 FP，逐步上采样 + 跳跃连接） =====
        # FP3: l3→l2，插值特征(1024) + 跳跃连接特征(512) = 1536 → 256
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        # FP2: l2→l1，插值特征(256) + 跳跃连接特征(320) = 576 → 128
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        # FP1: l1→l0，插值特征(128) + [类别one-hot(16) + 坐标(3) + 原始特征] = 150 → 128
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])

        # ===== 输出层 =====
        self.conv1 = nn.Conv1d(128, 128, 1)  # 1x1 卷积
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)  # 128 → 50（部件类别数）

    def forward(self, xyz, cls_label):
        """ 前向传播
        参数:
            xyz: [B, C, N] 输入点云（C=3 或 6）
            cls_label: [B, 16] 物体类别 one-hot（ShapeNetPart 16类物体）
        返回:
            x: [B, N, num_classes] 每个点的部件类别对数概率
            l3_points: [B, 1024] 全局特征
        """
        # ===== Set Abstraction 编码层（下采样） =====
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz          # 全部通道作为特征
            l0_xyz = xyz[:, :3, :]   # 前 3 维作为坐标
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,3,512], [B,320,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,3,128], [B,512,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,3,1], [B,1024,1]

        # ===== Feature Propagation 解码层（上采样 + 跳跃连接） =====
        # l3(1点) → l2(128点)：插值 l3 特征 + l2 原始特征
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 128]
        # l2(128点) → l1(512点)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 512]
        # 构造物体类别 one-hot，并复制到每个点
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)  # [B, 16, N]
        # l1(512点) → l0(N点)：拼接 [类别one-hot + 坐标 + l0原始特征] 作为跳跃连接
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)  # [B, 128, N]

        # ===== 输出层 =====
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # Conv1d + BN + ReLU
        x = self.drop1(feat)        # Dropout 防过拟合
        x = self.conv2(x)           # [B, num_classes, N]
        x = F.log_softmax(x, dim=1)  # 对每个点的类别做 log_softmax
        x = x.permute(0, 2, 1)       # [B, N, num_classes]
        return x, l3_points


class get_loss(nn.Module):
    """ 部件分割损失函数：逐点负对数似然损失 """

    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        """
        参数:
            pred: [B, N, num_classes] 预测的对数概率
            target: [B, N] 每个点的真实部件类别
            trans_feat: 转换特征（未使用）
        返回: 标量损失值
        """
        total_loss = F.nll_loss(pred, target)  # 逐点 NLL Loss
        return total_loss