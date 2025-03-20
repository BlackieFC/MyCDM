import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error
import numpy as np
from datetime import datetime
import torch.nn as nn

from utils.load_data import MyDataloader
from models.model import Baseline_FFT, Baseline_IRT_FFT  # 适配多种模型
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from tqdm.auto import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型训练参数配置')

    # 实验配置
    parser.add_argument('--proj_name', type=str, default='test_250319_00', help='项目名称，用于保存检查点')
    parser.add_argument('--data', type=str, default='NIPS34', choices=['NIPS34'], help='使用的数据集名称')
    parser.add_argument('--scenario', type=str, default='all', choices=['all', 'Algebra', 'Algebra_cold', 'GeometryandMeasure', 'Number', 'student_all', 'student_cut'], help='情景')

    # 训练超参数
    parser.add_argument('--epoch', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--bs', type=int, default=256, help='批次大小')
    parser.add_argument('--seed', type=int, default=43, help='随机种子')

    # 模型配置
    parser.add_argument('--bert_path', type=str, help='BERT预训练模型路径',
                        default='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/bert-base-uncased'            # BERT
                        # default='/mnt/new_pfs/liming_team/auroraX/songchentao/MyCDM/roberta/xlm-roberta-base'   # RoBERTa
                        # default='/mnt/new_pfs/liming_team/auroraX/LLM/bge-large-en-v1.5'                        # BGE
                        )
    parser.add_argument('--tau', type=float, default=0.1, help='温度系数')             # 仅对于u±模型的对比和正则化损失项生效
    parser.add_argument('--lambda_cl', type=int, default=0.5, help='对比损失权重')
    parser.add_argument('--lambda_reg', type=int, default=1.0, help='正则损失权重')

    # 训练控制
    parser.add_argument('-esp', '--early_stop_patience', type=int, default=5, help='早停等待轮数')
    parser.add_argument('-ckpt', '--checkpoint_dir', type=str, default=None,
                        help='检查点保存目录 (默认: /mnt/new_pfs/liming_team/auroraX/songchentao/MyCDM/checkpoints/{proj_name})')

    _args = parser.parse_args()
    if _args.checkpoint_dir is None:
        _args.checkpoint_dir = f'/mnt/new_pfs/liming_team/auroraX/songchentao/MyCDM/checkpoints/{_args.proj_name}'

    return _args


def main_parallel(args):

    # 初始化Accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=8,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    # 设置随机种子（确保多卡一致性）
    set_seed(args.seed)
    # 设备由accelerator自动管理
    device = accelerator.device

    # 数据路径配置（适配多种基座模型）
    data_root = f'../data/{args.data}/{args.scenario}'
    train_path = f'{data_root}/train.json'
    val_path = f'{data_root}/val.json'
    test_path = f'{data_root}/test.json'
    if 'roberta' in args.bert_path.lower():
        exer_tokens_path = f'{data_root}/exer_tokens_roberta.json'
    elif 'bge' in args.bert_path.lower():
        exer_tokens_path = f'{data_root}/exer_tokens_bge.json'
    elif 'bert' in args.bert_path.lower():
        exer_tokens_path = f'{data_root}/exer_tokens_bert.json'
    else:
        raise ValueError(f"不支持的模型类型: {args.bert_path}，请选择 'BERT', 'RoBERTa' 或 'BGE'")

    # 读取数据集三维
    with open(f'{data_root}/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    # # 单卡
    # best_model_path = f'{args.checkpoint_dir}/best_model.pt'
    # last_checkpoint_path = f'{args.checkpoint_dir}/last_checkpoint.pt'
    # # 创建检查点目录（默认为 f'./checkpoints/{proj_name}'）
    # os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 多卡
    best_model_path = f'{args.checkpoint_dir}/best_model/'
    last_checkpoint_path = f'{args.checkpoint_dir}/last_checkpoint/'
    # 创建检查点目录
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(last_checkpoint_path, exist_ok=True)

    # 初始化最佳验证集性能
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    early_stop_counter = 0
    start_epoch = 0

    """模型初始化（适配多种基座模型）"""
    # # 使用全量微调基础模型嵌入头 + MLP预报头
    # model = Baseline_FFT(num_students=student_n,
    #                      bert_model_name=args.bert_path,
    # #                      tau=args.tau,
    # #                      lambda_reg=args.lambda_reg,
    # #                      lambda_cl=args.lambda_cl,
    #                      ).to(device)
    # 使用全量微调基础模型嵌入头 + 三参数IRT预报头
    model = Baseline_IRT_FFT(num_students=student_n,
                                bert_model_name=args.bert_path,
                                # tau=args.tau,
                                # lambda_reg=args.lambda_reg,
                                # lambda_cl=args.lambda_cl,
                                ).to(device)

    # 数据加载器
    train_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=exer_tokens_path,
        data_set=train_path,
        offset=0,
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

    # 优化器配置（分段设置学习率）—— 无效化了命令行传参lr！！！！！！！
    bert_params = list(model.bert.parameters())     # 注意基座模型统一命名为了self.bert
    other_params = [p for p in model.parameters() if id(p) not in [id(bp) for bp in bert_params]]
    # 使用Adam优化器
    optimizer = torch.optim.Adam(
        [
            {"params": bert_params, "lr": 2e-5},    # 基座模型部分使用较小学习率
            {"params": other_params, "lr": 1e-3}    # 其他部分使用较大学习率
        ]
    )
    # # 使用AdamW优化器
    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": bert_params, "lr": 2e-5},
    #         {"params": other_params, "lr": 1e-3}
    #     ],
    #     weight_decay=0.01
    # )
      
    # Accelerate准备组件
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # 断点续训
    scheduler = None  # 无学习率调度
    try:
        if os.path.exists(last_checkpoint_path):
            print(f"发现检查点目录: {last_checkpoint_path}")
            # 使用accelerator加载状态，与保存方式匹配
            accelerator.load_state(last_checkpoint_path)
            
            # 加载额外保存的信息
            if os.path.exists(os.path.join(last_checkpoint_path, "early_stop_info.pt")):
                extra_info = torch.load(os.path.join(last_checkpoint_path, "early_stop_info.pt"))
                # 恢复训练状态
                start_epoch = extra_info['epoch'] + 1
                best_val_auc = extra_info['best_val_auc']
                early_stop_counter = extra_info['early_stop_counter']
                
                # 尝试加载best_val_loss (如果存在)
                best_val_loss = float('inf')
                if 'best_val_loss' in extra_info:
                    best_val_loss = extra_info['best_val_loss']
                
                # 尝试加载best_val_acc (如果存在)
                best_val_acc = 0
                if 'best_val_acc' in extra_info:
                    best_val_acc = extra_info['best_val_acc']
                    
                print(f"加载检查点：从epoch {start_epoch}恢复训练，当前最佳val_auc={best_val_auc:.4f}")
            else:
                print("未找到训练状态信息，将使用默认值")
                start_epoch = 0
                best_val_auc = 0
                best_val_loss = float('inf')
                best_val_acc = 0
                early_stop_counter = 0
        else:
            print("未找到检查点目录，将从头开始训练")
            start_epoch = 0
            best_val_auc = 0
            best_val_loss = float('inf')
            best_val_acc = 0
            early_stop_counter = 0
    except Exception as e:
        print(f"加载检查点时出错: {str(e)}，将从头开始训练")
        start_epoch = 0
        best_val_auc = 0
        best_val_loss = float('inf')
        best_val_acc = 0
        early_stop_counter = 0

    # 初始化wandb
    if accelerator.is_main_process:
        wandb_run_id = None
        # 如果是恢复训练，尝试从early_stop_info.pt获取wandb运行ID(如果存在)
        if start_epoch > 0 and os.path.exists(os.path.join(last_checkpoint_path, "early_stop_info.pt")):
            extra_info = torch.load(os.path.join(last_checkpoint_path, "early_stop_info.pt"))
            if 'wandb_run_id' in extra_info:
                wandb_run_id = extra_info['wandb_run_id']
        
        wandb.init(
            project=args.proj_name,
            config={**vars(args)},
            resume="allow" if start_epoch > 0 else None,
            id=wandb_run_id
        )
        # 将当前wandb运行ID保存至下一个检查点
        wandb_run_id = wandb.run.id

    # 训练循环
    for epoch in range(start_epoch, args.epoch):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.epoch}:")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}, training epoch {epoch + 1}")

        # 训练步骤 & 打印训练集性能（所有打印操作仅在主进程中进行）
        train_total_loss = train_parallel(accelerator, model, train_loader, optimizer)  # , args
        if accelerator.is_main_process:
            print(f"  Train Pred Loss: {train_total_loss:.4f}")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}, validating epoch {epoch + 1}")
            
        # 验证步骤 & 打印验证集性能
        val_pred_loss, val_acc, val_auc, _, _ = val_or_test_parallel(accelerator, model, val_loader)
        if accelerator.is_main_process:
            print(f"  Val Pred Loss: {val_pred_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}, testing epoch {epoch + 1}")
        
        # 测试步骤 & 打印测试集性能
        test_pred_loss, test_acc, test_auc, _, _ = val_or_test_parallel(accelerator, model, test_loader)
        if accelerator.is_main_process:
            print(f"  Test Pred Loss: {test_pred_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")

        # 记录日志（主进程）
        if accelerator.is_main_process:
            wandb.log({
                "train/pred_loss": train_total_loss,
                "val/pred_loss": val_pred_loss,
                "val/acc": val_acc,
                "val/auc": val_auc,
                "test/pred_loss": test_pred_loss,  # 250319新增：记录每轮的测试集性能
                "test/acc": test_acc,
                "test/auc": test_auc,
                "epoch": epoch + 1,
            })

        # 早停逻辑（主进程）
        if accelerator.is_main_process:
            # 每轮训练后将当前模型状态保存至last_checkpoint_path
            accelerator.save_state(last_checkpoint_path)
            
            # 额外保存当前轮次信息，用于可能的断点续训
            torch.save({
                'epoch': epoch,
                'current_val_auc': val_auc,
                'current_val_acc': val_acc,
                'current_val_loss': val_pred_loss,
                'best_val_auc': best_val_auc,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stop_counter,
                'wandb_run_id': wandb_run_id if 'wandb_run_id' in locals() else None
            }, os.path.join(last_checkpoint_path, "checkpoint_info.pt"))
            
            # 最佳模型保存逻辑
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_val_loss = val_pred_loss
                early_stop_counter = 0                
                
                # 使用accelerator.save_state()保存完整状态
                accelerator.save_state(best_model_path)
                
                # 额外保存一些自定义信息(可选)
                torch.save({
                    'epoch': epoch,
                    'best_val_auc': best_val_auc,
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                    'wandb_run_id': wandb_run_id if 'wandb_run_id' in locals() else None
                }, os.path.join(best_model_path, "metrics.pt"))
            else:
                early_stop_counter += 1
                
            # 触发早停（主进程）
            if early_stop_counter >= args.early_stop_patience:
                print(f"\n早停触发！连续{args.early_stop_patience}个epoch验证集无改进")
                
                # 保存最终状态
                accelerator.save_state(last_checkpoint_path)
                
                # 额外保存早停信息，同时包含所有需要的状态信息供断点续训使用
                torch.save({
                    'epoch': epoch,
                    'best_val_auc': best_val_auc,
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                    'early_stop_counter': early_stop_counter,
                    'wandb_run_id': wandb_run_id if 'wandb_run_id' in locals() else None
                }, os.path.join(last_checkpoint_path, "early_stop_info.pt"))
                break


    # 最终测试
    if os.path.exists(best_model_path):
        # 加载最佳模型 & 进行测试
        accelerator.load_state(os.path.dirname(best_model_path))
        test_pred_loss, test_acc, test_auc, y_pred, y_true = val_or_test_parallel(accelerator, model, test_loader)
        # 打印测试集性能（主进程）
        if accelerator.is_main_process:
            print("\n加载最佳模型进行测试...")
            print('========================================================================')
            print(f"\nFinal Test Results:")
            print(f"  Test Pred Loss: {test_pred_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")
            print('========================================================================')
        # 记录最终测试集性能（主进程）
        if accelerator.is_main_process:
            wandb.log({
                "final/test_loss": test_pred_loss,  # 250319修改：记录最终测试集性能
                "final/test_acc": test_acc,
                "final/test_auc": test_auc
            })
            wandb.finish()

    # 返回最佳验证集性能（不重要）
    return {
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_auc': best_val_auc
    }


def train_parallel(_accelerator, _model, _train_loader, _optimizer):
    _model.train()
    total_loss = 0.0
    progress_bar = tqdm(_train_loader, desc="Training", disable=not _accelerator.is_local_main_process)

    for batch in progress_bar:
        stu_ids = batch['stu_id']                  # 学生ID
        labels = batch['label'].float()            # 响应真实值
        input_ids = batch['input_ids']             # tokenize内容
        attention_mask = batch['attention_mask']   # 对应的mask
        exer_in = {'input_ids': input_ids,         # 组装为bert模型输入格式
                   'attention_mask': attention_mask
                   }

        # 梯度清零
        _optimizer.zero_grad()

        # 前向传播
        preds, _, _, _ = _model(stu_ids, exer_in)

        # 计算损失（显式调用损失函数）
        loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())

        # 反向传播
        _accelerator.backward(loss)

        # 更新参数
        _optimizer.step()

        # 累计损失
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/(progress_bar.n+1))

    # 计算平均损失
    avg_total_loss = total_loss / len(_train_loader)
    return avg_total_loss


def val_or_test_parallel(_accelerator, _model, _data_loader):
    _model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 所有GPU都会遍历完整的数据加载器（自动分片）
        for batch in _data_loader:
            stu_ids = batch['stu_id']
            labels = batch['label'].float()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            exer_in = {'input_ids': input_ids, 
                       'attention_mask': attention_mask
                       }

            # 前向传播
            output = _model(stu_ids, exer_in)
            preds = output[0]

            # 收集所有GPU的预测结果
            gathered_preds = _accelerator.gather(preds)
            gathered_labels = _accelerator.gather(labels.squeeze())
            
            # 主进程保存结果
            if _accelerator.is_main_process:
                all_preds.append(gathered_preds.cpu().numpy())
                all_labels.append(gathered_labels.cpu().numpy())

    # 只在主进程计算指标
    if _accelerator.is_main_process:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        binary_preds = (all_preds >= 0.5).astype(int)
        return (
            root_mean_squared_error(all_labels, all_preds),
            accuracy_score(all_labels, binary_preds),
            roc_auc_score(all_labels, all_preds),
            all_preds,
            all_labels
        )
    else:
        return None, None, None, None, None


if __name__ == '__main__':
    """
    accelerate launch --num_processes=4 --num_machines=1 --mixed_precision fp16 train_parallel.py --proj_name test_250319_00 --bs 256 --epoch 100 --scenario student_all
    """
    args_in = parse_args()
    main_parallel(args_in)