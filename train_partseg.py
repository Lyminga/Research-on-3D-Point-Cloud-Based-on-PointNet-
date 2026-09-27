"""
train_partseg.py —— PointNet++ 部件分割模型训练脚本
======================================================
功能:
  1. 加载 ShapeNetPart 数据集（16 大类，50 个部件类别）
  2. 构建 PointNet++ 部件分割模型（MSG/SSG）
  3. 训练 + 验证，保存最优模型（按 instance avg mIoU）
  4. 支持断点续训

运行示例:
  python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
"""

import argparse  # 命令行参数解析
import os  # 操作系统接口
from data_utils.ShapeNetDataLoader import PartNormalDataset  # ShapeNetPart 数据加载器
import torch  # PyTorch 框架
import datetime  # 日期时间
import logging  # 日志记录
from pathlib import Path  # 路径处理
import sys  # 系统接口
import importlib  # 动态导入模块
import shutil  # 文件复制
from tqdm import tqdm  # 进度条
import provider  # 数据增强工具
import numpy as np  # 数值计算
"""
训练所需设置参数：
--model pointnet2_part_seg_msg   模型名称
--normal                         是否使用法向
--log_dir pointnet2_part_seg_msg 日志目录名
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 把 models 目录加入 Python 路径

# 16 个大类各自包含的部件标签编号（共 50 个部件类别）
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
               'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
               'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
               'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table} 部件标签 → 所属大类
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 将标签转为 one-hot 编码
    输入: y: [B] 类别索引, num_classes: 类别数
    返回: [B, num_classes] one-hot 张量
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]  # 用单位矩阵取对应行
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    ''' 解析命令行参数 '''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name')  # 模型
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size [default: 4]')  # 批大小
    parser.add_argument('--epoch', default=5, type=int, help='Epoch to run [default: 5]')  # 训练轮数
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate [default: 0.001]')  # 学习率
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')  # GPU 编号
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')  # 优化器
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')  # 日志目录
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')  # 权重衰减
    parser.add_argument('--npoint', type=int, default=2048, help='Point Number [default: 2048]')  # 输入2048个点
    parser.add_argument('--normal', action='store_true', default=False, help='use normal info [default: False]')  # 法向开关
    parser.add_argument('--step_size', type=int, default=20, help='lr decay step [default: 20]')  # 学习率衰减步长
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')  # 学习率衰减率
    return parser.parse_args()


def main(args):
    def log_string(str):
        """ 同时写日志文件和打印到控制台 """
        logger.info(str)
        print(str)

    ''' 设置 GPU '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ''' 创建实验目录 '''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')  # 部件分割子目录
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')  # 模型权重目录
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')  # 日志目录
    log_dir.mkdir(exist_ok=True)

    ''' 配置日志 '''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    ''' 加载数据集 '''
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'  # 数据集路径
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval',
                                      normal_channel=args.normal)  # 训练+验证集
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test',
                                     normal_channel=args.normal)  # 测试集
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16  # 16 个大类
    num_part = 50  # 50 个部件类别

    ''' 加载模型 '''
    MODEL = importlib.import_module(args.model)  # 动态导入模型模块
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))  # 备份模型文件
    shutil.copy('models/pointnet_util.py', str(experiment_dir))  # 备份核心算子

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()  # 构建模型
    criterion = MODEL.get_loss().cuda()  # 损失函数

    def weights_init(m):
        """ Xavier 初始化权重 """
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # 断点续训
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)  # 应用 Xavier 初始化

    # 优化器
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        """ 调整 BatchNorm 的动量参数 """
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):  # 训练循环
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''调整学习率和 BN 动量'''
        # 学习率按 step_size 步长衰减，最低不低于 1e-5
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 应用新学习率
        mean_correct = []  # 存储每个 batch 的点级准确率
        # BN 动量随 epoch 衰减
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))  # 应用新动量

        '''训练一个 epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9, ascii=' >#'):
            points, label, target = data  # points:[B,N,6], label:[B,16], target:[B,N]
            points = points.data.numpy()  # 转 numpy 用于数据增强
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])  # 随机缩放
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])  # 随机平移
            points = torch.Tensor(points)  # 转回 tensor
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()  # 移到 GPU
            points = points.transpose(2, 1)  # [B, 6, N] channel first
            optimizer.zero_grad()  # 梯度清零
            classifier = classifier.train()  # 训练模式
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))  # 前向传播
            seg_pred = seg_pred.contiguous().view(-1, num_part)  # [B*N, 50]
            target = target.view(-1, 1)[:, 0]  # [B*N]
            pred_choice = seg_pred.data.max(1)[1]  # 取最大概率的类别
            correct = pred_choice.eq(target.data).cpu().sum()  # 统计正确点数
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))  # 点级准确率
            loss = criterion(seg_pred, target, trans_feat)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        train_instance_acc = np.mean(mean_correct)  # 平均训练准确率
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():  # 测试阶段，不计算梯度
            test_metrics = {}
            total_correct = 0  # 总正确点数
            total_seen = 0     # 总点数
            total_seen_class = [0 for _ in range(num_part)]  # 每个部件类别的点数
            total_correct_class = [0 for _ in range(num_part)]  # 每个部件类别正确点数
            shape_ious = {cat: [] for cat in seg_classes.keys()}  # 每个大类的 IoU 列表
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9, ascii=' >#'):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)  # [B, 6, N]
                classifier = classifier.eval()  # 评估模式
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))  # 前向传播
                cur_pred_val = seg_pred.cpu().data.numpy()  # [B, N, 50]
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # 最终预测结果
                target = target.cpu().data.numpy()
                for i in range(cur_batch_size):
                    # 根据真实标签确定所属大类，只在该大类的部件类别中取最大值
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(cur_pred_val == target)  # 统计正确点数
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):  # 按部件类别统计
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):  # 计算每个形状的 IoU
                    segp = cur_pred_val[i, :]  # 预测
                    segl = target[i, :]        # 真实
                    cat = seg_label_to_cat[segl[0]]  # 所属大类
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0  # 该部件不存在且预测也不存在，IoU=1
                        else:
                            # IoU = 交集 / 并集
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))  # 该形状各部件 IoU 均值

            all_shape_ious = []
            for cat in shape_ious.keys():  # 汇总所有形状的 IoU
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


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        # 保存最优模型（按 instance avg mIoU 判断）
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),  # 模型权重
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态（用于断点续训）
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # 更新最佳指标
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)

