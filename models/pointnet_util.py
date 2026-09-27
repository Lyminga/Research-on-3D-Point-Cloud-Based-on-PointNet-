"""
pointnet_util.py —— PointNet++ 核心算子库
==========================================
本文件实现了 PointNet++ 论文中的核心操作：
  1. 点云归一化 (pc_normalize)
  2. 距离计算 (square_distance)
  3. 按索引取点 (index_points)
  4. 最远点采样 FPS (farthest_point_sample)
  5. 球查询 Ball Query (query_ball_point)
  6. 采样+分组 (sample_and_group / sample_and_group_all)
  7. 集合抽象层 SSG (PointNetSetAbstraction)
  8. 集合抽象层 MSG (PointNetSetAbstractionMsg) —— 多尺度分组
  9. 特征传播层 FP (PointNetFeaturePropagation) —— 上采样插值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    # print("{}: {}s".format(tag, time() - t))  # 调试用：打印代码段耗时
    return time()

def pc_normalize(pc):
    """ 单个点云归一化：中心化 + 缩放到单位球
    输入: pc: Nx3 numpy 数组
    输出: 归一化后的 Nx3 数组
    """
    l = pc.shape[0]  # 点数 N
    centroid = np.mean(pc, axis=0)  # 计算质心
    pc = pc - centroid  # 中心化（减去质心）
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 最大半径
    pc = pc / m  # 缩放到单位球
    return pc

def square_distance(src, dst):
    """ 计算两组点之间的欧氏距离平方（向量化实现）
    原理: dist = ||src - dst||^2 = ||src||^2 + ||dst||^2 - 2*src·dst
    输入:
        src: [B, N, C] 源点
        dst: [B, M, C] 目标点
    输出:
        dist: [B, N, M] 每对源点和目标点之间的距离平方
    """
    B, N, _ = src.shape  # B=batch, N=源点数
    _, M, _ = dst.shape  # M=目标点数
    # -2 * src · dst^T  → [B, N, M]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # + ||src||^2  → [B, N, 1] 广播到 [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # + ||dst||^2  → [B, 1, M] 广播到 [B, N, M]
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """ 根据索引从点云中取出对应的点
    输入:
        points: [B, N, C] 输入点云
        idx: [B, S] 或 [B, S, K] 采样索引
    返回:
        new_points: [B, S, C] 或 [B, S, K, C] 取出的点
    """
    device = points.device
    B = points.shape[0]  # batch 大小
    view_shape = list(idx.shape)  # 索引的形状
    view_shape[1:] = [1] * (len(view_shape) - 1)  # 只保留 batch 维度
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # batch 维度不重复
    # 构造 batch 索引：[B, S] → 每个位置标注属于哪个 batch
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # 高级索引：points[batch索引, 点索引, :]
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """ 最远点采样 (FPS, Farthest Point Sampling)
    每次选择距离已选点集最远的点，保证采样点均匀覆盖整个点云。
    输入:
        xyz: [B, N, 3] 输入点云坐标
        npoint: 需要采样的点数
    返回:
        centroids: [B, npoint] 采样点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape  # B=batch, N=总点数, C=3
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)#8*512
    distance = torch.ones(B, N).to(device) * 1e10 #8*1024 每个点到已选点集的最近距离，初始化为很大
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#batch里每个样本随机初始化一个最远点的索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # batch 索引
    for i in range(npoint):  # 循环采样 npoint 次
        centroids[:, i] = farthest #第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)#得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)#计算当前采样点与其他点的距离
        mask = dist < distance#选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]#
        farthest = torch.max(distance, -1)[1]#重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """ 球查询 (Ball Query)：在每个中心点的 radius 半径内找 nsample 个邻居点
    输入:
        radius: 球的半径
        nsample: 每个球内最多采样的点数
        xyz: [B, N, 3] 所有点
        new_xyz: [B, S, 3] 查询中心点（FPS 采样得到的）
    返回:
        group_idx: [B, S, nsample] 每个中心点对应的邻居点索引
    """
    device = xyz.device
    B, N, C = xyz.shape  # N=总点数
    _, S, _ = new_xyz.shape  # S=中心点数
    # 初始化：每个中心的邻居索引为 0~N-1
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)#得到B N M （就是N个点中每一个和M中每一个的欧氏距离）
    group_idx[sqrdists > radius ** 2] = N #找到距离大于给定半径的设置成一个N值（1024）索引 表示不在半径当中
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#做升序排序，后面的都是大的值（1024）
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])#如果半径内的点没那么多，就直接用第一个点来代替了。。。
    mask = group_idx == N #返回布尔值，哪些是不在半径当中的点，哪些是不在半径当中的点的索引是N
    group_idx[mask] = group_first[mask] #将不在半径当中的点的索引设置为第一个点的索引进行填充
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """ 采样 + 分组：先用 FPS 采样中心点，再用 Ball Query 分组
    输入:
        npoint: 采样中心点数
        radius: 球查询半径
        nsample: 每个组内的点数
        xyz: [B, N, 3] 点坐标
        points: [B, N, D] 点特征（无法向则为 None）
    返回:
        new_xyz: [B, npoint, 3] 采样中心点坐标
        new_points: [B, npoint, nsample, 3+D] 分组后的点（相对坐标+特征）
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C] FPS 采样得到中心索引
    torch.cuda.empty_cache()  # 释放显存
    new_xyz = index_points(xyz, fps_idx)  # 取出中心点坐标 [B, npoint, 3]
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # 球查询得到邻居索引 [B, npoint, nsample]
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C] 取出邻居点坐标
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 相对坐标（去中心化）
    torch.cuda.empty_cache()

    if points is not None:  # 如果有额外特征（如法向量）
        grouped_points = index_points(points, idx)  # 取出邻居点特征
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D] 拼接相对坐标和特征
    else:
        new_points = grouped_xyz_norm
    if returnfps:  # 是否返回 FPS 索引（用于可视化）
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """ 全局分组：将所有点作为一个组（最后一层 SA 使用）
    输入:
        xyz: [B, N, 3]
        points: [B, N, D]
    返回:
        new_xyz: [B, 1, 3] 中心点（原点）
        new_points: [B, 1, N, 3+D] 所有点作为一组
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)  # 中心点设为原点
    grouped_xyz = xyz.view(B, 1, N, C)  # 所有点作为一组
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """ PointNet 集合抽象层（SSG, 单尺度分组）
    流程: FPS采样 → Ball Query分组 → 相对坐标+特征拼接 → MLP → MaxPooling
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        参数:
            npoint: FPS 采样点数
            radius: Ball Query 半径
            nsample: 每组点数
            in_channel: 输入特征通道数
            mlp: MLP 每层输出通道列表，如 [64, 64, 128]
            group_all: 是否全局分组（最后一层设为 True）
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()  # MLP 卷积层列表
        self.mlp_bns = nn.ModuleList()    # MLP 批归一化层列表
        last_channel = in_channel
        for out_channel in mlp:  # 逐层构建 MLP
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))  # 1x1 卷积
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        输入:
            xyz: [B, 3, N] 点坐标（channel first）
            points: [B, D, N] 点特征
        返回:
            new_xyz: [B, 3, S] 采样中心点坐标
            new_points: [B, D', S] 聚合后的特征
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3] 转为 channel last
        # print(xyz.shape) # 调试用：sa3时为 [8,128,3]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
        # print(points.shape) # 调试用：sa3时为 [8,128,640]
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)  # 全局分组
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)  # 采样分组
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # print(new_points.shape) # 调试用：sa3时当作一个组 [8batch,1组,128点,643特征]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint] 转回 channel first
        # print(new_points.shape) # 调试用：sa3时为 [8,643,128,1]
        for i, conv in enumerate(self.mlp_convs): #Conv2d(643,256,kernel_size=(1,1),stride=(1,1));Conv2d(256,512,kernel_size=(1,1),stride=(1,1));Conv2d(512,1024,kernel_size=(1,1),stride=(1,1))
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))  # Conv2d → BN → ReLU
        # print(new_points.shape) # 调试用：sa3时为 [8,1024,128,1]
        new_points = torch.max(new_points, 2)[0] #对一个组做pointnet的maxpooling操作
        # print(new_points.shape) # 调试用：sa3时为 [8,1024,1]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S] 转回 channel first
        # print(new_xyz.shape) # 调试用：sa3时为 [8,3,1]
        return new_xyz, new_points #返回位置信息和特征信息


class PointNetSetAbstractionMsg(nn.Module):
    """ PointNet 集合抽象层（MSG, 多尺度分组）
    与 SSG 的区别：使用多个不同半径的 Ball Query，分别提取特征后拼接。
    多尺度可以捕获不同粒度的局部结构，提升分割任务的精度。
    """

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """
        参数:
            npoint: FPS 采样点数
            radius_list: 多个球查询半径，如 [0.1, 0.2, 0.4]
            nsample_list: 每个半径对应的组内点数，如 [16, 32, 128]
            in_channel: 输入特征通道数
            mlp_list: 每个尺度对应的 MLP 通道列表，如 [[32,32,64], [64,64,128], [64,96,128]]
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()  # 每个尺度的卷积块
        self.bn_blocks = nn.ModuleList()    # 每个尺度的 BN 块
        for i in range(len(mlp_list)):  # 为每个尺度构建 MLP
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # 输入特征 + 相对坐标(3)
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        输入:
            xyz: [B, 3, N]
            points: [B, D, N]
        返回:
            new_xyz: [B, 3, S]
            new_points_concat: [B, D', S] 多尺度特征拼接
        """
        xyz = xyz.permute(0, 2, 1) #就是坐标点位置特征
        # print(xyz.shape) # 调试用
        if points is not None:
            points = points.permute(0, 2, 1) ##就是额外提取的特征，第一次的时候就是那个法向量特征
        # print(points.shape) # 调试用
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))#采样后的点
        # print(new_xyz.shape) # 调试用
        new_points_list = []  # 存储各尺度的特征
        for i, radius in enumerate(self.radius_list):  # 遍历每个尺度
            K = self.nsample_list[i]  # 当前尺度的组内点数
            group_idx = query_ball_point(radius, K, xyz, new_xyz)#返回的是索引
            grouped_xyz = index_points(xyz, group_idx)#得到各个组中实际点（通过索引找实际值）
            grouped_xyz -= new_xyz.view(B, S, 1, C)#去mean new_xyz相当于簇的中心点（去均值）
            if points is not None: # pointnet就是缺少点和点的关系，pointnet++这里需要将点和点的特征拼接起来提取特征
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                # print(grouped_points.shape) # 调试用：把点的法向量特征和点的位置特征拼接起来，变成D+3 [8,512,16,6]；在sa2中的第一次变为[8,128组,32点,323特征] 3+320 xyz加上第一次得到的320个特征
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S] [8,6,16,512]；在sa2中的第一次变为[8,323,32,128]
            # print(grouped_points.shape) # 调试用：channel first原则做了个提前操作
            for j in range(len(self.conv_blocks[i])):  # 当前尺度的 MLP
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            # print(grouped_points.shape) # 调试用：做了3次卷积，特征维度变大了 [8,64,16,512](仅第一个半径数值举例)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S] 就是pointnet里的maxpool操作 [8,64,512] [8,128,512] [8,128,512]
            # print(new_points.shape) # 调试用
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]
        new_points_concat = torch.cat(new_points_list, dim=1)  # 拼接各尺度特征
        # print(new_points_concat.shape) # 调试用：拼接在一起 [8,320,512] 64+128+128=320 (r=0.4 128;r=0.2 128;r=0.1 64)；在sa2中变为[8,640,128](128+256+256)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """ 特征传播层 (FP, Feature Propagation)
    用于分割任务的上采样：将稀疏点的特征插值传播到稠密点。
    方法：对每个目标点，找其在源点中最近的 3 个点，用距离倒数加权插值。
    """

    def __init__(self, in_channel, mlp):
        """
        参数:
            in_channel: 拼接后的输入通道数（跳跃连接特征 + 插值特征）
            mlp: MLP 通道列表
        """
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))  # 1D 卷积
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        输入:
            xyz1: [B, 3, N] 稠密点坐标（插值目标）
            xyz2: [B, 3, S] 稀疏点坐标（插值来源）
            points1: [B, D, N] 稠密点特征（跳跃连接）
            points2: [B, D, S] 稀疏点特征
        返回:
            new_points: [B, D', N] 上采样后的特征
        """
        xyz1 = xyz1.permute(0, 2, 1)  #dubug断点可以打在这里  [B, N, 3]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, 3]
        # print(xyz1.shape) # 调试用：fp3:[4,128,3];fp2:[4,513,3];fp1:[4,2048,3]
        # print(xyz2.shape) # 调试用：fp3:[4,1,3];fp2:[4,128,3]现在是128个点变512个点，不是复制了，而是根据距离加权插值;fp1:[4,512,3]

        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        # print(points2.shape) # 调试用：fp3:[4,1,1024];fp2:[4,128,256];fp1:[4,512,128]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:  # 只有一个点时直接复制（最后一层 FP）
            interpolated_points = points2.repeat(1, N, 1) #fp3的复制操作
            # print(interpolated_points.shape) # 调试用：[4,128,1024]
        else:
            dists = square_distance(xyz1, xyz2)  # [B, N, S] 稠密点到稀疏点的距离
            # print(dists.shape) # 调试用：fp2:[4,512,128];fp1:[4,2048,512]
            dists, idx = dists.sort(dim=-1)  # 按距离升序排序
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3] 取最近的 3 个点

            dist_recip = 1.0 / (dists + 1e-8)  # 距离倒数作为权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 权重归一化分母
            weight = dist_recip / norm  # 归一化权重
            # print(weight.shape) # 调试用：fp2:[4,512,3];fp1:[4,2048,3]
            # print(index_points(points2, idx).shape) # 调试用：fp2:[4,512,3,256];fp1:[4,2048,3,128]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # 加权求和
            # print(interpolated_points.shape) # 调试用：fp2:[4,512,256];fp1:[4,2048,128]

        if points1 is not None:  # 拼接跳跃连接特征
            points1 = points1.permute(0, 2, 1)  # [B, N, D]
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D+D']
        else:
            new_points = interpolated_points
        # print(new_points.shape) # 调试用：拼接在一起 fp3:[4,128,1536] 1024+512; fp2:[4,512,576] 320+256;fp1:[4,2048,150]
        new_points = new_points.permute(0, 2, 1) #channel first原则做了个提前操作
        # print(new_points.shape) # 调试用：fp3:[4,1536,128]; fp2:[4,576,512];fp1:[4,150,2048]
        for i, conv in enumerate(self.mlp_convs):  # MLP 处理
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # print(new_points.shape) # 调试用：fp3后：[4,256特征,128个点]; fp2后：[4,128特征,512个点];fp1后：[4,128特征,2048个点] 之后就对这128维特征分出50个组件
        return new_points

