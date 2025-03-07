import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error
import numpy as np
from datetime import datetime

from utils.load_data import MyDataloader
from models.model import MyCDM_MLP_FFT
from transformers import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='模型训练参数配置')

    # 实验配置
    parser.add_argument('--proj_name', type=str, default='freeze_250221_00', help='项目名称，用于保存检查点')
    parser.add_argument('--data', type=str, default='NIPS34', choices=['NIPS34'], help='使用的数据集名称')
    parser.add_argument('--scenario', type=str, default='all', choices=['all', 'Algebra', 'Algebra_cold', 'GeometryandMeasure', 'Number'], help='情景')
    # parser.add_argument('--mode', choices=['baseline', 'freeze', 'fine-tune'], default='freeze', help='实验模式')

    # 训练超参数
    parser.add_argument('--epoch', type=int, default=100, help='最大训练轮数')
    # parser.add_argument('--bs', type=int, default=256, help='批次大小')
    # parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='学习率')

    # 模型配置
    parser.add_argument('--bert_path', type=str, help='BERT预训练模型路径',
                        default='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/bert-base-uncased')
    parser.add_argument('--tau', type=float, default=0.1, help='温度系数')
    parser.add_argument('--lambda_cl', type=int, default=0.5, help='对比损失权重')
    parser.add_argument('--lambda_reg', type=int, default=0.1, help='正则损失权重')

    # 训练控制
    parser.add_argument('-esp', '--early_stop_patience', type=int, default=5, help='早停等待轮数')
    parser.add_argument('-ckpt', '--checkpoint_dir', type=str, default=None,
                        help='检查点保存目录 (默认: ../checkpoints/{proj_name})')
    # parser.add_argument('--grid_search', action='store_true', help='格点搜索调参')
    # parser.add_argument('--verbose', type=int, default=0, help='是否显示epoch内进度')

    _args = parser.parse_args()
    # 默认ckpt路径
    if _args.checkpoint_dir is None:
        _args.checkpoint_dir = f'../checkpoints/{_args.proj_name}'

    return _args


def main_parallel(args):

    # <editor-fold desc="初始化accelerator和device管理（核心修改）">
    accelerator = Accelerator(
        mixed_precision='fp16',                         # 从参数读取或固定为'fp16'
        gradient_accumulation_steps=8,                  # args.grad_accum_steps,
        deepspeed_plugin=args.deepspeed_plugin_config   # 此处会自动从配置文件读取
    )
    # 设置随机种子（确保多卡一致性）
    set_seed(args.seed)
    # 设备由accelerator自动管理
    device = accelerator.device
    # </editor-fold>

    # <editor-fold desc="数据路径配置">
    data_root = f'../data/{args.data}/{args.scenario}'
    train_path = f'{data_root}/train.json'
    val_path = f'{data_root}/val.json'
    test_path = f'{data_root}/test.json'
    exer_tokens_path = f'{data_root}/exer_tokens.json'

    # 读取数据配置
    with open(f'{data_root}/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    # 创建检查点目录（默认为 f'./checkpoints/{proj_name}'）
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = f'{args.checkpoint_dir}/best_model.pt'
    last_checkpoint_path = f'{args.checkpoint_dir}/last_checkpoint.pt'

    # 初始化训练状态
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    early_stop_counter = 0
    start_epoch = 0
    # </editor-fold>

    # <editor-fold desc="封装模型、优化器、dataloader">
    # 声明模型
    model = MyCDM_MLP_FFT(
        num_students=student_n,
        bert_model_name=args.bert_path,
        tau=args.tau,
        lambda_reg=args.lambda_reg,
        lambda_cl=args.lambda_cl
    )

    # 设置Dataloader
    train_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=exer_tokens_path,  # None or path( of json)
        data_set=train_path,
        offset=0,  # 使用原始ID的数据集
        pid_zero=0
    )
    val_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=exer_tokens_path,
        data_set=val_path,
        offset=0,
        pid_zero=0
    )
    test_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=exer_tokens_path,
        data_set=test_path,
        offset=0,
        pid_zero=0
    )

    # 创建两组参数：BERT和非BERT
    bert_params = set(model.bert.parameters())
    other_params = [p for p in model.parameters() if p not in bert_params]
    # 设置优化器
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": 2e-5},    # BERT使用较小学习率
            {"params": other_params, "lr": 1e-3}                # CDM较大学习率
        ],
        weight_decay=0.01
    )
    # 使用accelerator准备组件（关键修改）
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        [model, optimizer, train_loader, val_loader, test_loader])
    # # 从模型中获取DeepSpeed自动创建的优化器
    # optimizer = model.optimizer  # 此时优化器参数由ds_config定义（如lr=1e-3）
    # </editor-fold>

    # <editor-fold desc="断点续训 & wandb并行修改">
    # 断点续训
    if os.path.exists(last_checkpoint_path):
        # （并行）加载模型和优化器
        checkpoint = torch.load(last_checkpoint_path, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # 覆盖相应的训练状态记录参数，打印
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        early_stop_counter = checkpoint['early_stop_counter']
        print(f"加载检查点：从epoch {start_epoch}恢复训练，当前最佳val_loss={best_val_loss:.4f}")
    # 初始化wandb
    if os.path.exists(last_checkpoint_path):
        wandb.init(
            project=args.proj_name,
            config={**vars(args)},
            resume=True if start_epoch > 0 else False,
            reinit=True if args.grid_search else False  # 是否允许重复初始化
        )
    # </editor-fold>

    # <editor-fold desc="训练循环并行化">
    for epoch in range(start_epoch, args.epoch):
        # （0）所有print操作均只在主进程上进行（下同）
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.epoch}:")
            now = datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"), f', training epoch {epoch + 1}')

        # （1）训练 & 可视化
        train_total_loss, train_pred_loss, train_cl_loss, train_reg_loss = train_parallel(accelerator, model, train_loader, optimizer)

        """后续过程均在主进程中进行"""
        if accelerator.is_main_process:
            print(f"  Train Pred Loss: {train_pred_loss:.4f}, total Loss: {train_total_loss:.4f}, CL Loss: {train_cl_loss:.4f}, Reg Loss: {train_reg_loss:.4f} ")
            now = datetime.now()
            print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, validating epoch {epoch + 1}')
            # （2）验证 & 可视化
            val_pred_loss, val_acc, val_auc, _, _ = val_or_test_parallel(accelerator, model, val_loader)
            print(f"  Val Pred Loss: {val_pred_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
            now = datetime.now()
            print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, testing epoch {epoch + 1}')
            # （3）测试 & 可视化
            test_pred_loss, test_acc, test_auc, _, _ = val_or_test_parallel(accelerator, model, test_loader)
            print(f"  Test Pred Loss: {test_pred_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")

            # （4）早停逻辑（AUC优先）
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_val_loss = val_pred_loss
                early_stop_counter = 0
                # 保存模型（使用accelerator接口）
                accelerator.save({
                    'epoch': epoch,
                    'model_state': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_auc': val_auc
                }, best_model_path)
            else:
                early_stop_counter += 1

            # （5）记录训练和验证指标
            wandb.log({
                "train/total_loss": train_total_loss,
                "train/pred_loss": train_pred_loss,
                "train/cl_loss": train_cl_loss,
                "train/reg_loss": train_reg_loss,
                "val/pred_loss": val_pred_loss,
                "val/acc": val_acc,
                "val/auc": val_auc,
                "epoch": epoch + 1,
                # "val/best_loss": best_val_loss,
                # "early_stop_counter": early_stop_counter
            })

            # （6）保存最新检查点（用于断点续训）
            if early_stop_counter >= args.early_stop_patience:
                print(f"\n早停触发！连续{args.early_stop_patience}个epoch验证集无改进")
                accelerator.save({
                    'epoch': epoch,
                    'model_state': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_auc': best_val_auc,
                    'early_stop_counter': early_stop_counter
                }, last_checkpoint_path)
                break



    # 最终测试（加载最佳模型）
    if accelerator.is_main_process:
        if os.path.exists(best_model_path):
            print("\n加载最佳模型进行测试...")
            accelerator.load_state(best_model_path)
        else:
            print("未找到最优模型ckpt！")

        print('========================================================================')
        now = datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, testing...')
        test_pred_loss, test_acc, test_auc, y_pred, y_true = val_or_test_parallel(accelerator, model, test_loader)
        print(f"\nFinal Test Results:")
        print(f"  Test Pred Loss: {test_pred_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")

        now = datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, finish.')
        print('========================================================================')

        # 保存真实值和预报值
        pass

        # 记录测试结果
        wandb.log({
            "test/pred_loss": test_pred_loss,
            "test/acc": test_acc,
            "test/auc": test_auc
        })

        # 记录最终结果
        wandb.finish()

    # </editor-fold>

    # 返回验证集指标用于网格搜索比较
    return {
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_auc': best_val_auc
    }


def train_parallel(_accelerator, _model, _train_loader, _optimizer):  # , _device, verbose=0
    """
    并行训练函数
    """
    # device、模型状态、初始化loss计数、进度条
    _model.train()
    total_loss = 0.0
    pred_loss = 0.0
    cl_loss = 0.0
    reg_loss = 0.0
    progress_bar = tqdm(_train_loader, desc="Training", disable=not _accelerator.is_local_main_process)
    _device = _accelerator.device

    # count = 0
    for batch in progress_bar:
        # # 进度可视化（改为使用进度条）
        # if _accelerator.is_main_process and verbose and (count + 1) % 200 == 0:
        #     _now = datetime.now()
        #     print(f'{_now.strftime("%Y-%m-%d %H:%M:%S")}, {count+1} of {len(_train_loader)}')
        # count += 1

        # 处理batch数据，组装为bert模型输入格式
        stu_ids = batch['stu_id'].to(_device)                 # 学生ID
        labels = batch['label'].to(_device).float()           # 响应真实值
        input_ids = batch['input_ids'].to(_device)            # tokenize内容
        attention_mask = batch['attention_mask'].to(_device)  # 对应的mask
        exer_in = {'input_ids': input_ids, 'attention_mask': attention_mask}  # 组装为bert模型输入格式

        _optimizer.zero_grad()                                # 梯度清零
        output = _model(stu_ids, exer_in)                     # 前向传播
        loss, loss_bce, loss_cl, loss_reg  = _model.get_loss(output, labels)
        _accelerator.backward(loss)                           # 反向传播
        _optimizer.step()                                     # 优化器调整权重

        total_loss += loss.item()                             # 累计损失
        pred_loss += loss_bce.item()
        cl_loss += loss_cl.item()
        reg_loss += loss_reg.item()
        progress_bar.set_postfix(
            loss=total_loss / (progress_bar.n + 1),           # 进度条可视化
            bce=pred_loss / (progress_bar.n + 1),
            cl=cl_loss / (progress_bar.n + 1),
            reg=reg_loss / (progress_bar.n + 1)
        )

    # 计算该轮训练的最终平均损失
    avg_total_loss = total_loss / len(_train_loader)
    avg_pred_loss = pred_loss / len(_train_loader)
    avg_cl_loss = cl_loss / len(_train_loader)
    avg_reg_loss = reg_loss / len(_train_loader)

    # 返回各项损失
    return avg_total_loss, avg_pred_loss, avg_cl_loss, avg_reg_loss


def val_or_test_parallel(_accelerator, _model, _data_loader):  # , _device, verbose=0
    """
    并行验证or测试函数
    """
    # device、模型状态、初始化loss计数、进度条
    _model.eval()
    pred_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in _data_loader:
            # 处理batch数据
            stu_ids = batch['stu_id']                                             # 学生ID
            labels = batch['label'].float()                                       # 响应真实值
            input_ids = batch['input_ids']                                        # tokenize内容
            attention_mask = batch['attention_mask']                              # 对应的mask
            exer_in = {'input_ids': input_ids, 'attention_mask': attention_mask}  # 组装为bert模型输入格式

            # 前向传播 & 收集结果
            output = _model(stu_ids, exer_in)                                     # 前向传播
            preds = output[0]                                                     # 获取预测概率
            all_preds.append(_accelerator.gather(preds))                          # 收集所有设备的预测结果
            all_labels.append(_accelerator.gather(labels))

    # 合并结果
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 只在主进程计算指标
    if _accelerator.is_main_process:
        all_preds = all_preds.detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy()
        binary_preds = (all_preds >= 0.5).astype(int)  # 二值化预测结果
        acc = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)
        rmse = root_mean_squared_error(all_labels, all_preds)
        return rmse, acc, auc, all_preds, all_labels
    else:
        return None, None, None



if __name__ == '__main__':

    args_in = parse_args()
    main_parallel(args_in)
