"""
test_partseg.py —— PointNet++ 部件分割测试 + 可视化脚本
========================================================
功能:
  1. 加载训练好的部件分割模型
  2. 在测试集上评估（支持投票 num_votes 提升精度）
  3. 输出 Accuracy / Class avg IoU / Instance avg IoU
  4. 可视化：各类别 mIoU 柱状图 + 点云分割对比图
"""
import argparse  # 命令行参数解析
import os  # 操作系统接口
from data_utils.ShapeNetDataLoader import PartNormalDataset  # ShapeNetPart 数据加载器
import torch  # PyTorch 框架
import logging  # 日志
import sys  # 系统接口
import importlib  # 动态导入模型
from tqdm import tqdm  # 进度条
import numpy as np  # 数值计算
import matplotlib
matplotlib.use('Agg')  # 后台模式，不弹出窗口
import matplotlib.pyplot as plt  # 绘图
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 把 models 加入路径

# 16 个大类各自包含的部件标签编号（共 50 个部件类别）
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table} 部件标签→大类
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 标签转 one-hot 编码 """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing [default: 24]')  # 批大小
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')  # GPU
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')  # 点数
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_ssg', help='Experiment root')  # 实验目录
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')  # 法向
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')  # 投票次数
    return parser.parse_args()

def main(args):
    def log_string(str):
        """ 同时写日志和打印 """
        logger.info(str)
        print(str)

    '''设置 GPU'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir  # 实验目录

    '''配置日志'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'  # 数据集路径

    # 加载测试集
    TEST_DATASET = PartNormalDataset(root = root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 16  # 16 个大类
    num_part = 50  # 50 个部件类别

    '''加载模型'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]  # 从日志文件名获取模型名
    MODEL = importlib.import_module(model_name)  # 动态导入模型
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()  # 构建模型
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  # 加载最优权重
    classifier.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():  # 测试阶段，不计算梯度
        test_metrics = {}
        total_correct = 0  # 总正确点数
        total_seen = 0     # 总点数
        total_seen_class = [0 for _ in range(num_part)]  # 各部件类别点数
        total_correct_class = [0 for _ in range(num_part)]  # 各部件类别正确点数
        shape_ious = {cat: [] for cat in seg_classes.keys()}  # 各大类 IoU 列表
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        vis_samples = []  # 收集前几个样本用于点云可视化
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9, ascii=' >#'):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)  # [B, 6, N]
            classifier = classifier.eval()  # 评估模式
            # 投票机制：对同一个样本预测 num_votes 次取平均，提升鲁棒性
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred
            seg_pred = vote_pool / args.num_votes  # 平均投票结果
            cur_pred_val = seg_pred.cpu().data.numpy()  # [B, N, 50]
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # 最终预测
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                # 根据真实标签确定大类，只在该大类部件类别中取最大
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)  # 统计正确点数
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):  # 按部件类别统计
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            # 收集前 6 个不同类别的样本用于可视化
            if len(vis_samples) < 6:
                for i in range(cur_batch_size):
                    if len(vis_samples) >= 6:
                        break
                    cat_name = seg_label_to_cat[target[i, 0]]
                    # 每个类别只取一个样本
                    if not any(s['cat'] == cat_name for s in vis_samples):
                        pts = points[i, :3, :].cpu().numpy().T  # (N, 3) 点坐标
                        vis_samples.append({
                            'cat': cat_name,
                            'points': pts,
                            'pred': cur_pred_val[i, :],
                            'gt': target[i, :]
                        })
            # 计算每个形状的 IoU
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]  # 预测
                segl = target[i, :]        # 真实
                cat = seg_label_to_cat[segl[0]]  # 所属大类
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # 该部件不存在且预测也不存在
                        part_ious[l - seg_classes[cat][0]] = 1.0  # IoU=1
                    else:
                        # IoU = 交集 / 并集
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))  # 该形状各部件 IoU 均值

        # 汇总所有 IoU
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])  # 该大类平均 IoU
        mean_shape_ious = np.mean(list(shape_ious.values()))  # 所有大类平均 IoU
        test_metrics['accuracy'] = total_correct / float(total_seen)  # 点级准确率
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))  # 类别平均准确率
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious  # 类别平均 IoU
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)  # 实例平均 IoU


    # 打印最终结果
    log_string('Accuracy is: %.5f'%test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f'%test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f'%test_metrics['inctance_avg_iou'])

    '''可视化：各类别 mIoU 柱状图 + 点云分割结果对比'''
    cats_sorted = sorted(seg_classes.keys())
    ious_sorted = [shape_ious[cat] for cat in cats_sorted]

    # ---- 1. 各类别 mIoU 柱状图 ----
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#2ecc71' if iou >= 0.8 else '#f39c12' if iou >= 0.5 else '#e74c3c'
              for iou in ious_sorted]
    bars = ax.bar(range(len(cats_sorted)), [i * 100 for i in ious_sorted],
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(cats_sorted)))
    ax.set_xticklabels(cats_sorted, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Per-Class mIoU  (Class Avg mIoU: %.2f%%, Instance Avg mIoU: %.2f%%)'
                 % (test_metrics['class_avg_iou'] * 100, test_metrics['inctance_avg_iou'] * 100))
    ax.set_ylim(0, 105)
    ax.axhline(y=test_metrics['class_avg_iou'] * 100, color='blue', linestyle='--', linewidth=1,
               label='Class Avg = %.1f%%' % (test_metrics['class_avg_iou'] * 100))
    ax.legend()
    for bar, iou in zip(bars, ious_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                '%.0f' % (iou * 100), ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    iou_path = os.path.join(experiment_dir, 'per_class_iou.png')
    fig.savefig(iou_path, dpi=120)
    plt.close(fig)
    log_string('各类别 mIoU 图已保存: %s' % iou_path)

    # ---- 2. 点云分割结果对比（真实 vs 预测） ----
    n_samples = len(vis_samples)
    if n_samples > 0:
        fig = plt.figure(figsize=(16, 3 * n_samples))
        for idx, sample in enumerate(vis_samples):
            pts = sample['points']
            gt = sample['gt']
            pred = sample['pred']
            # 左：真实标签
            ax1 = fig.add_subplot(n_samples, 2, idx * 2 + 1, projection='3d')
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=gt, s=1, cmap='tab20', vmin=0, vmax=49)
            ax1.set_title('%s - Ground Truth' % sample['cat'], fontsize=10)
            ax1.set_axis_off()
            # 右：预测标签
            ax2 = fig.add_subplot(n_samples, 2, idx * 2 + 2, projection='3d')
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pred, s=1, cmap='tab20', vmin=0, vmax=49)
            ax2.set_title('%s - Prediction' % sample['cat'], fontsize=10)
            ax2.set_axis_off()
        fig.suptitle('Part Segmentation Results (GT vs Prediction)', fontsize=14, y=1.0)
        fig.tight_layout()
        seg_path = os.path.join(experiment_dir, 'seg_visualization.png')
        fig.savefig(seg_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        log_string('点云分割对比图已保存: %s' % seg_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)

