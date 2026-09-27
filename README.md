# PointNet++  PyTorch 实现

本项目是 PointNet++（PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space）的 PyTorch 实现，支持三大任务：

- **分类（Classification）**：ModelNet40，40 类物体
- **部件分割（Part Segmentation）**：ShapeNetPart，16 类物体 / 50 个部件
- **语义分割（Semantic Segmentation）**：S3DIS 室内场景（需额外数据）

---

## 一、项目结构

```
Pointnet2/
├── models/                      # 模型定义
│   ├── pointnet_util.py         # 【核心】FPS / Ball Query / SA / FP 算子
│   ├── pointnet2_cls_msg.py     # 分类模型（MSG 多尺度分组）
│   ├── pointnet2_cls_ssg.py     # 分类模型（SSG 单尺度分组）
│   ├── pointnet2_part_seg_msg.py# 部件分割模型（MSG）
│   ├── pointnet2_part_seg_ssg.py# 部件分割模型（SSG）
│   ├── pointnet2_sem_seg_msg.py # 语义分割模型（MSG）
│   ├── pointnet2_sem_seg.py     # 语义分割模型（SSG）
│   ├── pointnet_cls.py          # PointNet 分类（原版）
│   ├── pointnet_part_seg.py     # PointNet 部件分割（原版）
│   ├── pointnet_sem_seg.py      # PointNet 语义分割（原版）
│   └── pointnet.py              # PointNet 工具函数（T-Net 等）
│
├── data_utils/                  # 数据加载
│   ├── ModelNetDataLoader.py    # ModelNet40 数据加载
│   ├── ShapeNetDataLoader.py    # ShapeNetPart 数据加载
│   ├── S3DISDataLoader.py       # S3DIS 数据加载
│   ├── collect_indoor3d_data.py # S3DIS 数据预处理
│   └── indoor3d_util.py         # S3DIS 工具函数
│
├── data/                        # 数据集（需自行下载）
│   ├── modelnet40_normal_resampled/   # ModelNet40
│   ├── shapenetcore_partanno_segmentation_benchmark_v0_normal/  # ShapeNetPart
│   └── stanford_indoor3d/              # S3DIS（语义分割需要）
│
├── provider.py                  # 数据增强（随机缩放、平移、旋转、抖动）
├── train_cls.py                 # 分类训练脚本
├── test_cls.py                  # 分类测试 + 可视化
├── train_partseg.py             # 部件分割训练脚本
├── test_partseg.py              # 部件分割测试 + 可视化
├── train_semseg.py              # 语义分割训练脚本
├── test_semseg.py               # 语义分割测试脚本
└── log/                         # 实验日志与模型权重（自动生成）
    ├── classification/
    │   └── pointnet2_cls_msg/
    │       ├── checkpoints/best_model.pth
    │       ├── logs/
    │       ├── pointnet_util.py        # 训练时自动备份
    │       └── pointnet2_cls_msg.py    # 训练时自动备份
    └── part_seg/
        └── pointnet2_part_seg_msg/
            ├── checkpoints/best_model.pth
            ├── logs/
            ├── pointnet_util.py
            └── pointnet2_part_seg_msg.py
```

---

## 二、核心算子说明（pointnet_util.py）

| 算子 | 功能 |
|------|------|
| `farthest_point_sample(xyz, npoint)` | 最远点采样（FPS）：均匀采样 npoint 个中心点 |
| `query_ball_point(radius, nsample, xyz, new_xyz)` | Ball Query：以中心点为球心，半径内取 nsample 个邻居 |
| `sample_and_group(npoint, radius, nsample, xyz, points)` | 采样 + 分组：FPS + Ball Query + 相对坐标拼接 |
| `sample_and_group_all(xyz, points)` | 全局分组：所有点作为一组（最后一层 SA 使用） |
| `PointNetSetAbstraction` | 集合抽象层（SSG）：单尺度分组 → MLP → MaxPooling |
| `PointNetSetAbstractionMsg` | 集合抽象层（MSG）：多尺度分组 → 各自 MLP → 特征拼接 |
| `PointNetFeaturePropagation` | 特征传播层（FP）：最近 3 点距离倒数加权插值上采样 |

### 网络结构对比

**分类网络（pointnet2_cls_msg.py）**：
```
输入 [B, 6, N]
  → SA1(MSG, 512点, 3半径) → 320维
  → SA2(MSG, 128点, 3半径) → 640维
  → SA3(全局分组) → 1024维
  → FC(1024→512) → Dropout → FC(512→256) → Dropout → FC(256→40)
输出 [B, 40] 各类别概率
```

**部件分割网络（pointnet2_part_seg_msg.py）**（U-Net 风格编解码）：
```
输入 [B, 6, N] + 类别 one-hot [B, 16]
  编码: SA1(512点,320维) → SA2(128点,512维) → SA3(全局,1024维)
  解码: FP3(l3→l2) → FP2(l2→l1) → FP1(l1→l0, 拼接类别one-hot)
  → Conv1d(128→50)
输出 [B, N, 50] 每个点的部件类别概率
```

---

## 三、环境配置

### 依赖版本

- Python 3.6+
- PyTorch 1.12.0+cu113
- matplotlib（可视化）
- numpy、tqdm

### conda 环境（本项目使用 pointnet2）

```bash
conda create -n pointnet2 python=3.9
conda activate pointnet2
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tqdm numpy
```

---

## 四、数据集准备

### 1. ModelNet40（分类）

下载地址：https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

解压到：`data/modelnet40_normal_resampled/`

目录结构：
```
data/modelnet40_normal_resampled/
├── modelnet40_shape_names.txt
├── modelnet40_train.txt
├── modelnet40_test.txt
├── airplane/
│   ├── airplane_0001.txt   # 每行: x y z nx ny nz
│   └── ...
└── ...
```

### 2. ShapeNetPart（部件分割）

下载地址：https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

解压到：`data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`

### 3. S3DIS（语义分割，可选）

下载地址需申请：http://buildingparser.stanford.edu/dataset.html

解压到：`data/stanford_indoor3d/`，然后运行 `data_utils/collect_indoor3d_data.py` 预处理。

---

## 五、运行指南

### 1. 分类任务

#### 训练

```bash
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
```

- `--model`：模型名（`pointnet2_cls_msg` 或 `pointnet2_cls_ssg`）
- `--normal`：使用法向量（输入 6 通道）
- `--log_dir`：日志目录名
- `--epoch`：训练轮数（默认 5，已调小用于快速验证）
- `--batch_size`：批大小（默认 24）

#### 测试 + 可视化

```bash
python test_cls.py --log_dir pointnet2_cls_msg --normal
```

输出：
- 控制台：Test Instance Accuracy / Class Accuracy
- `log/classification/pointnet2_cls_msg/confusion_matrix.png`：混淆矩阵
- `log/classification/pointnet2_cls_msg/per_class_accuracy.png`：各类别准确率柱状图

**实测结果**：Test Instance Accuracy 80.62%，Class Accuracy 70.76%（5 epoch）

---

### 2. 部件分割任务

#### 训练

```bash
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
```

- `--epoch`：默认 5（已调小）
- `--batch_size`：默认 4（显存不足可减小）

训练日志输出：
- Train accuracy
- 每个大类的 mIoU
- Class avg mIoU / Instance avg mIoU
- 最优模型保存至 `log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth`

#### 测试 + 可视化

```bash
python test_partseg.py --log_dir pointnet2_part_seg_msg --normal
```

- `--num_votes`：投票次数（默认 3，提升鲁棒性）

输出：
- Accuracy / Class avg accuracy / Class avg mIoU / Instance avg mIoU
- `log/part_seg/pointnet2_part_seg_msg/per_class_iou.png`：各类别 mIoU 柱状图
- `log/part_seg/pointnet2_part_seg_msg/seg_visualization.png`：点云分割对比图（真实 vs 预测）

---

### 3. 语义分割任务（需 S3DIS 数据）

```bash
python train_semseg.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg
```

---

## 六、数据增强（provider.py）

训练时对输入点云做以下增强，提升模型泛化能力：

| 函数 | 作用 |
|------|------|
| `random_scale_point_cloud` | 随机缩放（0.8 ~ 1.25 倍） |
| `shift_point_cloud` | 随机平移（±0.1） |
| `rotate_perturbation_point_cloud` | 随机小角度旋转 |
| `jitter_point_cloud` | 高斯噪声抖动 |
| `random_point_dropout` | 随机丢弃部分点 |
| `shift_point_cloud` | 随机平移 |

---

## 七、可视化说明

所有可视化图片保存在对应实验目录下：

| 图片 | 路径 | 说明 |
|------|------|------|
| 混淆矩阵 | `log/classification/<log_dir>/confusion_matrix.png` | 40×40 类别混淆矩阵 |
| 类别准确率 | `log/classification/<log_dir>/per_class_accuracy.png` | 40 类准确率柱状图 |
| 各类别 mIoU | `log/part_seg/<log_dir>/per_class_iou.png` | 16 大类 mIoU 柱状图 |
| 分割对比 | `log/part_seg/<log_dir>/seg_visualization.png` | 点云真实/预测对比图 |

---

## 八、训练技巧

1. **tqdm 进度条**：已统一添加 `ascii='>#'`，避免 Windows 终端 Unicode 乱码
2. **epoch 调小**：默认 5 epoch 可快速跑通，正式训练建议 200+ epoch
3. **法向量**：分类任务建议使用 `--normal`，输入 6 通道（xyz + 法向）
4. **断点续训**：若 `checkpoints/best_model.pth` 存在，自动加载继续训练
5. **投票测试**：部件分割测试时 `--num_votes` 设为 3，多次预测取平均提升精度

---

## 九、参考文献

- Qi, C. R., et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." NeurIPS 2017.
- 原始实现：https://github.com/charlesq34/pointnet2
- PyTorch 实现参考：https://github.com/yanx27/Pointnet_Pointnet2_pytorch
