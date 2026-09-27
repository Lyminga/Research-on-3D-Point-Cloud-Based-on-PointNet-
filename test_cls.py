"""
test_cls.py —— PointNet++ 分类模型测试脚本
==============================================
功能:
  1. 加载训练好的模型权重
  2. 在 ModelNet40 测试集上评估（支持多次投票提升精度）
  3. 生成混淆矩阵和各类别准确率可视化图

运行示例:
  python test_cls.py --normal --log_dir pointnet2_cls_msg --num_votes 3
"""

from data_utils.ModelNetDataLoader import ModelNetDataLoader  # ModelNet40 数据加载器
import argparse  # 命令行参数解析
import numpy as np  # 数值计算
import os  # 操作系统接口
import torch  # PyTorch 框架
import logging  # 日志记录
from tqdm import tqdm  # 进度条
import sys  # 系统接口
import importlib  # 动态导入模块
import matplotlib
matplotlib.use('Agg')  # 后台模式，不弹出窗口
import matplotlib.pyplot as plt  # 绘图库

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 把 models 目录加入 Python 路径

"""
配置参数：
--normal                    是否使用法向
--log_dir pointnet2_cls_msg 实验目录名（与训练时一致）
"""

def parse_args():
    ''' 解析命令行参数 '''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')  # 批大小
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')  # GPU 编号
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')  # 输入点数
    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_normal', help='Experiment root')  # 实验目录
    parser.add_argument('--normal', action='store_true', default=True, help='use normal info [default: False]')  # 法向开关
    parser.add_argument('--num_votes', type=int, default=3, help='voting times [default: 3]')  # 投票次数
    return parser.parse_args()

def test(model, loader, num_class=40, vote_num=1):
    """ 测试函数：在测试集上评估模型（支持多次投票）
    参数:
        model: 模型
        loader: 测试数据加载器
        num_class: 类别数（40）
        vote_num: 投票次数（对同一样本多次预测取平均）
    返回:
        instance_acc: 实例准确率
        class_acc: 类别平均准确率
        all_preds: 所有预测标签（用于混淆矩阵）
        all_targets: 所有真实标签（用于混淆矩阵）
        per_class_acc: 每个类别的准确率（用于柱状图）
    """
    mean_correct = []  # 每个 batch 的准确率
    class_acc = np.zeros((num_class, 3))  # [正确预测数, 出现次数, 该类准确率]
    all_preds = []      # 收集所有预测标签（用于混淆矩阵）
    all_targets = []    # 收集所有真实标签（用于混淆矩阵）
    for j, data in tqdm(enumerate(loader), total=len(loader), ascii=' >#'):  # 遍历测试集
        points, target = data
        target = target[:, 0]  # 取出标签 [B]
        points = points.transpose(2, 1)  # [B, C, N] channel first
        points, target = points.cuda(), target.cuda()  # 移到 GPU
        classifier = model.eval()  # 评估模式
        # 投票：对同一批数据预测 vote_num 次，取概率平均
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()  # 累加预测概率
        for _ in range(vote_num):
            pred, _ = classifier(points)  # 前向传播
            vote_pool += pred  # 累加概率
        pred = vote_pool / vote_num  # 平均概率
        pred_choice = pred.data.max(1)[1]  # 取最大概率的类别
        # 记录预测和真实标签，用于后续可视化
        all_preds.extend(pred_choice.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        for cat in np.unique(target.cpu()):  # 对每个出现的类别
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()  # 该类正确数
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])  # 累加该类准确率
            class_acc[cat, 1] += 1  # 该类出现次数 +1
        correct = pred_choice.eq(target.long().data).cpu().sum()  # 当前 batch 正确数
        mean_correct.append(correct.item() / float(points.size()[0]))  # 记录 batch 准确率
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]  # 每个类别的平均准确率
    per_class_acc = class_acc[:, 2]  # 每个类别的准确率（用于柱状图）
    class_acc = np.mean(class_acc[:, 2])  # 所有类别平均
    instance_acc = np.mean(mean_correct)  # 实例准确率
    return instance_acc, class_acc, np.array(all_preds), np.array(all_targets), per_class_acc


def main(args):
    def log_string(str):
        """ 同时写日志文件和打印到控制台 """
        logger.info(str)
        print(str)

    ''' 设置 GPU '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ''' 实验目录 '''
    experiment_dir = 'log/classification/' + args.log_dir  # 训练时保存的目录

    ''' 配置日志 '''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  # 日志写入 eval.txt
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    ''' 加载测试数据集 '''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'  # 数据集路径
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                      normal_channel=args.normal)  # 测试集
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)  # 测试加载器

    ''' 加载模型 '''
    num_class = 40  # ModelNet40 共 40 类
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]  # 从 logs 目录获取模型名
    MODEL = importlib.import_module(model_name)  # 动态导入模型模块

    classifier = MODEL.get_model(num_class, normal_channel=args.normal).cuda()  # 构建模型

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  # 加载最优权重
    classifier.load_state_dict(checkpoint['model_state_dict'])  # 载入权重

    # 读取类别名称（40类）
    shape_names = [line.strip() for line in open(
        os.path.join(DATA_PATH, 'modelnet40_shape_names.txt'))]

    with torch.no_grad():  # 不计算梯度，节省显存
        instance_acc, class_acc, all_preds, all_targets, per_class_acc = test(
            classifier.eval(), testDataLoader, vote_num=args.num_votes)  # 执行测试
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

    '''可视化：混淆矩阵 + 各类别准确率'''
    # ---- 1. 混淆矩阵（用 numpy 实现，无需 sklearn） ----
    # 混淆矩阵 cm[i][j] 表示真实类别 i 被预测为类别 j 的样本数
    cm = np.zeros((num_class, num_class), dtype=int)  # 初始化 40x40 全零矩阵
    for t, p in zip(all_targets, all_preds):  # 遍历所有样本
        cm[t, p] += 1  # 对应位置 +1
    fig, ax = plt.subplots(figsize=(18, 16))  # 创建画布
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 用蓝色系热力图显示
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # 添加颜色条
    ax.set(xticks=np.arange(num_class), yticks=np.arange(num_class),
           xticklabels=shape_names, yticklabels=shape_names,  # 设置轴标签为类别名
           title='Confusion Matrix (Instance Acc: %.2f%%)' % (instance_acc * 100),
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)  # x 轴标签旋转 90 度
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()  # 自动调整布局
    cm_path = os.path.join(experiment_dir, 'confusion_matrix.png')  # 保存路径
    fig.savefig(cm_path, dpi=120)  # 保存图片
    plt.close(fig)  # 关闭画布释放内存
    log_string('混淆矩阵已保存: %s' % cm_path)

    # ---- 2. 各类别准确率柱状图 ----
    fig, ax = plt.subplots(figsize=(18, 6))  # 创建画布
    # 根据准确率给柱子上色：≥80%绿色，50-80%橙色，<50%红色
    colors = ['#2ecc71' if acc >= 0.8 else '#f39c12' if acc >= 0.5 else '#e74c3c'
              for acc in per_class_acc]
    bars = ax.bar(range(num_class), per_class_acc * 100, color=colors, edgecolor='black', linewidth=0.5)  # 画柱状图
    ax.set_xticks(range(num_class))
    ax.set_xticklabels(shape_names, rotation=90, fontsize=8)  # x 轴为类别名
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy (Class Avg: %.2f%%)' % (class_acc * 100))
    ax.set_ylim(0, 105)  # y 轴范围 0-105
    ax.axhline(y=class_acc * 100, color='blue', linestyle='--', linewidth=1,  # 画类别平均准确率虚线
               label='Class Avg = %.1f%%' % (class_acc * 100))
    ax.legend()
    # 在柱子上方标注数值
    for bar, acc in zip(bars, per_class_acc):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    '%.0f' % (acc * 100), ha='center', va='bottom', fontsize=6)
    fig.tight_layout()
    acc_path = os.path.join(experiment_dir, 'per_class_accuracy.png')
    fig.savefig(acc_path, dpi=120)
    plt.close(fig)
    log_string('各类别准确率图已保存: %s' % acc_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)
