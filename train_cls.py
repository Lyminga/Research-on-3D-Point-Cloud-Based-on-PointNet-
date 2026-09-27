"""
train_cls.py —— PointNet++ 分类模型训练脚本
==============================================
功能:
  1. 加载 ModelNet40 数据集
  2. 构建 PointNet++ 分类模型（MSG/SSG）
  3. 训练 + 验证，保存最优模型
  4. 支持断点续训

运行示例:
  python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
"""

from data_utils.ModelNetDataLoader import ModelNetDataLoader  # ModelNet40 数据加载器
import argparse  # 命令行参数解析
import numpy as np  # 数值计算
import os  # 操作系统接口
import torch  # PyTorch 框架
import datetime  # 日期时间
import logging  # 日志记录
from pathlib import Path  # 路径处理
from tqdm import tqdm  # 进度条
import sys  # 系统接口
import provider  # 数据增强工具
import importlib  # 动态导入模块
import shutil  # 文件复制

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 把 models 目录加入 Python 路径
"""
需要配置的参数:
--model pointnet2_cls_msg   模型名称
--normal                    是否使用法向
--log_dir pointnet2_cls_msg 日志目录名
"""


def parse_args():
    ''' 解析命令行参数 '''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training [default: 8]')  # 批大小
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet2_cls_msg]')  # 模型
    parser.add_argument('--epoch', default=5, type=int, help='number of epoch in training [default: 5]')  # 训练轮数
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate [default: 0.001]')  # 学习率
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')  # GPU 编号
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')  # 输入点数
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer [default: Adam]')  # 优化器
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')  # 实验目录名
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')  # 权重衰减
    parser.add_argument('--normal', action='store_true', default=False, help='use normal info [default: False]')  # 法向开关
    return parser.parse_args()


def test(model, loader, num_class=40):
    """ 测试函数：在测试集上评估模型
    返回: instance_acc（实例准确率）, class_acc（类别平均准确率）
    """
    mean_correct = []  # 每个 batch 的准确率
    class_acc = np.zeros((num_class, 3))  # [正确预测数, 出现次数, 该类准确率]
    for j, data in tqdm(enumerate(loader), total=len(loader), ascii=' >#'):  # 遍历测试集
        points, target = data
        target = target[:, 0]  # 取出标签 [B]
        points = points.transpose(2, 1)  # [B, C, N] channel first
        points, target = points.cuda(), target.cuda()  # 移到 GPU
        classifier = model.eval()  # 切换到评估模式
        pred, _ = classifier(points)  # 前向传播
        pred_choice = pred.data.max(1)[1]  # 取最大概率的类别索引
        for cat in np.unique(target.cpu()):  # 对每个出现的类别
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()  # 该类正确数
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])  # 累加该类准确率
            class_acc[cat, 1] += 1  # 该类出现次数 +1
        correct = pred_choice.eq(target.long().data).cpu().sum()  # 当前 batch 正确数
        mean_correct.append(correct.item() / float(points.size()[0]))  # 记录 batch 准确率
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]  # 每个类别的平均准确率
    class_acc = np.mean(class_acc[:, 2])  # 所有类别平均
    instance_acc = np.mean(mean_correct)  # 实例准确率（所有样本平均）
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        """ 同时写日志文件和打印到控制台 """
        logger.info(str)
        print(str)

    ''' 设置 GPU '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ''' 创建实验目录 '''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))  # 当前时间字符串
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)  # 创建 log 目录
    experiment_dir = experiment_dir.joinpath('classification')  # 分类子目录
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)  # 用时间命名
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)  # 用指定名称
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')  # 模型权重目录
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')  # 日志目录
    log_dir.mkdir(exist_ok=True)

    ''' 配置日志 '''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))  # 日志文件
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    ''' 加载数据集 '''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'  # 数据集路径（硬编码）
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                       normal_channel=args.normal)  # 训练集
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                      normal_channel=args.normal)  # 测试集
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)  # 训练数据加载器
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)  # 测试数据加载器

    ''' 加载模型 '''
    num_class = 40  # ModelNet40 共 40 类
    MODEL = importlib.import_module(args.model)  # 动态导入模型模块
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))  # 备份模型文件
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))  # 备份核心算子

    classifier = MODEL.get_model(num_class, normal_channel=args.normal).cuda()  # 构建模型并移到 GPU
    criterion = MODEL.get_loss().cuda()  # 损失函数

    ''' 断点续训：加载已有 checkpoint '''
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']  # 从上次的 epoch 继续
        classifier.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    ''' 优化器 '''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # 每 20 epoch 学习率 ×0.7
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0  # 最佳实例准确率
    best_class_acc = 0.0     # 最佳类别准确率
    mean_correct = []

    ''' 开始训练 '''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()  # 更新学习率
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                   smoothing=0.9, ascii=' >#'):
            points, target = data
            points = points.data.numpy()
            # 数据增强：随机丢弃点 → 随机缩放 → 随机平移
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]  # [B]

            points = points.transpose(2, 1)  # [B, C, N]
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()  # 梯度清零

            classifier = classifier.train()  # 训练模式
            pred, trans_feat = classifier(points)  # 前向传播
            loss = criterion(pred, target.long(), trans_feat)  # 计算损失
            pred_choice = pred.data.max(1)[1]  # 预测类别
            correct = pred_choice.eq(target.long().data).cpu().sum()  # 正确数
            mean_correct.append(correct.item() / float(points.size()[0]))  # batch 准确率
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            global_step += 1

            # 每 50 个 batch 打印一次实时进度
            if (batch_id + 1) % 50 == 0:
                cur_loss = loss.item()
                cur_acc = correct.item() / float(points.size()[0])
                print('[Epoch %d/%d] batch %d/%d | loss: %.4f | acc: %.4f' % (
                    epoch + 1, args.epoch, batch_id + 1, len(trainDataLoader), cur_loss, cur_acc), flush=True)

        train_instance_acc = np.mean(mean_correct)  # 训练集准确率
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # 每个 epoch 结束后在测试集上评估
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            # 保存最优模型
            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
