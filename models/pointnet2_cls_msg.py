"""
pointnet2_cls_msg.py —— PointNet++ 分类模型（MSG 多尺度分组）
============================================================
网络结构:
  输入: [B, 6, N] (xyz 3通道 + 法向 3通道) 或 [B, 3, N] (仅 xyz)
  SA1 (MSG): 512点，3个半径(0.1,0.2,0.4)，输出 320 维特征
  SA2 (MSG): 128点，3个半径(0.2,0.4,0.8)，输出 640 维特征
  SA3 (全局): 所有点为一组，输出 1024 维全局特征
  FC1(1024→512) → Dropout → FC2(512→256) → Dropout → FC3(256→40)
  输出: [B, 40] 各类别的对数概率
"""

import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    """ PointNet++ 分类模型主体 """

    def __init__(self, num_class, normal_channel=True):
        """
        参数:
            num_class: 类别数（ModelNet40 为 40）
            normal_channel: 是否使用法向量（True 则输入通道为 6，否则为 3）
        """
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0  # 输入法向通道数（3维法向）
        self.normal_channel = normal_channel

        # 第一层集合抽象（MSG，多尺度）：FPS采512点，3个半径，输出 64+128+128=320 维
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # 第二层集合抽象（MSG，多尺度）：FPS采128点，输出 128+256+256=640 维
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # 第三层集合抽象（全局分组）：所有点为一组，输出 1024 维全局特征
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # 全连接分类头
        self.fc1 = nn.Linear(1024, 512)  # 1024 → 512
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)  # 丢弃 40% 神经元防过拟合
        self.fc2 = nn.Linear(512, 256)  # 512 → 256
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)  # 丢弃 50%
        self.fc3 = nn.Linear(256, num_class)  # 256 → 40（类别数）

    def forward(self, xyz):
        """ 前向传播
        输入: xyz: [B, 6, N]（含法向）或 [B, 3, N]（仅坐标）
        输出: x: [B, num_class] 各类别对数概率, l3_points: [B, 1024] 全局特征
        """
        B, _, _ = xyz.shape #断点可以打在这  B=batch 大小
        if self.normal_channel:
            norm = xyz[:, 3:, :]  # 取出法向部分 [B, 3, N]
            xyz = xyz[:, :3, :]   # 取出坐标部分 [B, 3, N]
        else:
            norm = None
        # print(xyz.shape)  # 调试用：打印输入点云坐标形状 [B, 3, N]
        # print(norm.shape)  # 调试用：打印输入法向形状 [B, 3, N]
        # 逐层前向传播
        l1_xyz, l1_points = self.sa1(xyz, norm)        # [B, 3, 512], [B, 320, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 128], [B, 640, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 1], [B, 1024, 1]
        # 展平全局特征 [B, 1024, 1] → [B, 1024]
        x = l3_points.view(B, 1024)
        # 三层全连接 + BN + ReLU + Dropout
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))   # [B, 512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))   # [B, 256]
        x = self.fc3(x)                                  # [B, 40]
        x = F.log_softmax(x, -1) #分类任务 前面3个整合为1024个特征对应于每个样本，对每个样本连上三个全连接层，输出得到40个值，完成40个类别的分类

        return x, l3_points


class get_loss(nn.Module):
    """ 分类损失函数：负对数似然损失 (NLL Loss) """

    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        """
        参数:
            pred: [B, num_class] 模型输出的对数概率
            target: [B] 真实类别标签
            trans_feat: 转换特征（分类任务未使用，分割任务用于正则化）
        返回: 标量损失值
        """
        total_loss = F.nll_loss(pred, target)  # 负对数似然损失
        return total_loss


