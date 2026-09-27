"""
provider.py —— 点云数据增强工具库
====================================
本文件提供了一系列点云数据增强函数，用于训练时扩充数据集、提升模型泛化能力。
主要包括：归一化、随机打乱、随机旋转、随机缩放、随机平移、随机抖动、随机丢弃点等。
所有函数的输入都是 numpy 数组，输出也是 numpy 数组。
"""

import numpy as np  # 数值计算库


# ============================================================
# 1. 归一化：将每个点云的中心移到原点，并缩放到单位球内
# ============================================================
def normalize_data(batch_data):
    """ 归一化批量点云数据：中心化 + 缩放至单位球
    输入:
        batch_data: BxNxC 数组 (B=batch大小, N=点数, C=通道数)
    输出:
        BxNxC 归一化后的数组
    """
    B, N, C = batch_data.shape  # 获取批量大小、点数、通道数
    normal_data = np.zeros((B, N, C))  # 创建与输入同形状的全零数组
    for b in range(B):  # 遍历每一个点云
        pc = batch_data[b]  # 取出第 b 个点云，形状 NxC
        centroid = np.mean(pc, axis=0)  # 计算点云的质心（所有点的平均值）
        pc = pc - centroid  # 减去质心，使点云中心平移到原点
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 计算所有点到原点的最远距离（最大半径）
        pc = pc / m  # 除以最大半径，将点云缩放到单位球内（最远点距离=1）
        normal_data[b] = pc  # 保存归一化后的点云
    return normal_data


# ============================================================
# 2. 打乱数据顺序
# ============================================================
def shuffle_data(data, labels):
    """ 随机打乱数据和标签的顺序（保持数据与标签的对应关系）
    输入:
        data: B,N,... numpy 数组
        labels: B,... numpy 数组
    返回:
        打乱后的数据、标签、以及打乱用的索引
    """
    idx = np.arange(len(labels))  # 生成 0 到 B-1 的索引数组
    np.random.shuffle(idx)  # 随机打乱索引
    return data[idx, ...], labels[idx], idx  # 用打乱的索引重排数据和标签


def shuffle_points(batch_data):
    """ 打乱每个点云内部的点的顺序（会影响 FPS 采样结果，作为一种数据增强）
        整个 batch 使用同一个打乱索引
    输入:
        batch_data: BxNxC 数组
    输出:
        BxNxC 打乱点序后的数组
    """
    idx = np.arange(batch_data.shape[1])  # 生成 0 到 N-1 的点索引
    np.random.shuffle(idx)  # 随机打乱点索引
    return batch_data[:, idx, :]  # 所有 batch 用同一索引重排点的顺序


# ============================================================
# 3. 随机旋转（绕 Y 轴 / 绕 Z 轴 / 带法向旋转）
# ============================================================
def rotate_point_cloud(batch_data):
    """ 随机绕 Y 轴（上方向）旋转点云，用于数据增强
    输入:
        batch_data: BxNx3 数组
    返回:
        BxNx3 旋转后的数组
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 创建输出数组
    for k in range(batch_data.shape[0]):  # 遍历每个点云
        rotation_angle = np.random.uniform() * 2 * np.pi  # 随机生成 0~2π 的旋转角
        cosval = np.cos(rotation_angle)  # 旋转角的余弦
        sinval = np.sin(rotation_angle)  # 旋转角的正弦
        # 构建绕 Y 轴的 3x3 旋转矩阵
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]  # 取出第 k 个点云
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 点乘旋转矩阵
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """ 随机绕 Z 轴旋转点云（与上面类似，只是旋转轴不同）
    输入: BxNx3，返回: BxNx3
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        # 构建绕 Z 轴的旋转矩阵
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    """ 随机绕 Y 轴旋转点云及其法向量（坐标和法向量要一起旋转）
    输入:
        batch_xyz_normal: BxNx6，前 3 通道是 XYZ 坐标，后 3 通道是法向量
    输出: BxNx6 旋转后的坐标和法向量
    """
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]  # 取出坐标部分
        shape_normal = batch_xyz_normal[k, :, 3:6]  # 取出法向量部分
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 旋转坐标
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)  # 旋转法向量
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ 随机小角度扰动旋转点云（绕 X、Y、Z 三轴各旋转一个小角度）
        与上面的大角度随机旋转不同，这里是小角度扰动，模拟轻微的姿态变化
    输入:
        batch_data: BxNx6（坐标+法向）
        angle_sigma: 高斯噪声的标准差
        angle_clip: 角度裁剪上限
    返回: BxNx6
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # 生成 3 个轴的小旋转角，并用 clip 限制范围
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        # 分别构建绕 X、Y、Z 轴的旋转矩阵
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))  # 组合三个旋转矩阵
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ 按指定角度绕 Y 轴旋转点云（角度由外部传入，不是随机的）
    输入: BxNx3，rotation_angle: 指定旋转角
    返回: BxNx3
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ 按指定角度绕 Y 轴旋转点云及其法向量
    输入: BxNx6，rotation_angle: 指定旋转角
    返回: BxNx6
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ 随机小角度扰动旋转点云（只旋转坐标，不含法向量）
    输入: BxNx3，返回: BxNx3
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


# ============================================================
# 4. 随机抖动（给每个点加微小噪声）
# ============================================================
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ 对每个点添加高斯噪声（模拟传感器噪声）
    输入:
        batch_data: BxNx3
        sigma: 高斯噪声标准差
        clip: 噪声裁剪范围
    返回: BxNx3
    """
    B, N, C = batch_data.shape
    assert (clip > 0)  # 确保裁剪范围为正
    # 生成高斯噪声并裁剪到 [-clip, clip] 范围
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data  # 将噪声加到原始点云上
    return jittered_data


# ============================================================
# 5. 随机平移
# ============================================================
def shift_point_cloud(batch_data, shift_range=0.1):
    """ 随机平移整个点云（每个点云整体加一个小偏移）
        随机的平移（加或者减）
    输入: BxNx3，shift_range: 平移范围
    返回: BxNx3
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))  # 每个点云生成一个 3D 平移向量
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]  # 整体平移
    return batch_data


# ============================================================
# 6. 随机缩放
# ============================================================
def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ 随机缩放整个点云（每个点云乘以一个随机缩放因子）
        各个点的数值大小随机改变[0.8到1.125之间随机扩大或缩小]
    输入: BxNx3，scale_low/scale_high: 缩放范围
    返回: BxNx3
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)  # 每个点云生成一个缩放因子
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]  # 整体缩放
    return batch_data


# ============================================================
# 7. 随机丢弃点（模拟点云稀疏化）
# ============================================================
def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """ 随机丢弃点云中的部分点（用第一个点替代被丢弃的点）
        batch_pc: BxNx3 对batch中每一个数据选取一部分点来去掉（用第一个点来替代）
    输入: batch_pc: BxNx3，max_dropout_ratio: 最大丢弃比例
    返回: BxNx3
    """
    for b in range(batch_pc.shape[0]):  # 遍历每个点云
        dropout_ratio = np.random.random() * max_dropout_ratio  # 随机生成丢弃比例 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]  # 选出要丢弃的点的索引
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # 用第一个点的坐标替代被丢弃的点
    return batch_pc
