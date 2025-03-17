import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime
import itertools
import json
from pathlib import Path

from utils.load_data import MyDataloader
from models.model import Baseline_IRT, Baseline_MLP, MyCDM_MLP, IRT, MyCDM_MSA, MyCDM_IRT, MyCDM_MLP_FFT, Basiline_MLP_FFT, Baseline_FFT, Baseline_IRT_FFT
from tqdm.auto import tqdm


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='模型训练参数配置')

    # 实验配置
    parser.add_argument('--mode', choices=['baseline', 'freeze', 'fine-tune'], default='freeze', help='实验模式')
    parser.add_argument('--proj_name', type=str, default='freeze_250221_00', help='项目名称，用于保存检查点')
    parser.add_argument('--data', type=str, default='NIPS34', choices=['NIPS34'], help='使用的数据集名称')
    parser.add_argument('--scenario', type=str, default='all', choices=['all', 'Algebra', 'Algebra_cold', 'GeometryandMeasure', 'Number', 'student_all', 'student_cut'], help='情景')

    # 训练超参数
    parser.add_argument('--bs', type=int, default=512, help='批次大小')
    parser.add_argument('--epoch', type=int, default=100, help='最大训练轮数')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='学习率')

    # 模型配置
    parser.add_argument('--bert_path', type=str, help='BERT预训练模型路径',
                        # default='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/bert-base-uncased'          # BERT
                        default='/mnt/new_pfs/liming_team/auroraX/songchentao/MyCDM/roberta/xlm-roberta-base'   # RoBERTa
                        # default='/mnt/new_pfs/liming_team/auroraX/LLM/bge-large-en-v1.5'                        # BGE
                        )
    parser.add_argument('--tau', type=float, default=0.1, help='温度系数')
    parser.add_argument('--lambda_cl', type=int, default=0.5, help='对比损失权重')
    parser.add_argument('--lambda_reg', type=int, default=1.0, help='正则损失权重')

    # 训练控制
    parser.add_argument('--grid_search', action='store_true', help='格点搜索调参')
    parser.add_argument('-esp', '--early_stop_patience', type=int, default=5, help='早停等待轮数')
    parser.add_argument('-ckpt', '--checkpoint_dir', type=str, default=None, help='检查点保存目录 (默认: ../checkpoints/{proj_name})')
    parser.add_argument('--verbose', type=int, default=0, help='是否显示epoch内进度')

    _args = parser.parse_args()

    # 后处理依赖参数
    if _args.checkpoint_dir is None:
        _args.checkpoint_dir = f'../checkpoints/{_args.proj_name}'

    return _args


def train(_model, _train_loader, _optimizer, _device, mode='baseline', verbose=0):
    """
    模型训练函数
        mode=['baseline','freeze','fine-tune']
    """
    _model.train()
    total_loss = 0.0
    pred_loss = 0.0
    cl_loss = 0.0
    reg_loss = 0.0

    progress_bar = tqdm(_train_loader, desc="Training")

    for batch in progress_bar:
        # 数据准备
        stu_ids = batch['stu_id'].to(_device)                     # 学生ID
        labels = batch['label'].to(_device).float()               # 响应真实值
        if mode == 'fine-tune':
            input_ids = batch['input_ids'].to(_device)            # tokenize内容
            attention_mask = batch['attention_mask'].to(_device)  # 对应的mask
            # 组装为bert模型输入格式
            exer_in = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            exer_in = batch['exer_id'].to(_device)                # 题目ID

        # 梯度清零
        _optimizer.zero_grad()

        # 前向传播
        output = _model(stu_ids, exer_in)
        loss, loss_bce, loss_cl, loss_reg  = _model.get_loss(output, labels)

        # 反向传播
        loss.backward()
        _optimizer.step()

        # 累计损失
        total_loss += loss.item()
        pred_loss += loss_bce.item()
        if mode in ['freeze','fine-tune']:
            cl_loss += loss_cl.item()
            reg_loss += loss_reg.item()

    # 计算平均损失
    avg_total_loss = total_loss / len(_train_loader)
    avg_pred_loss = pred_loss / len(_train_loader)
    avg_cl_loss = cl_loss / len(_train_loader)
    avg_reg_loss = reg_loss / len(_train_loader)

    return avg_total_loss, avg_pred_loss, avg_cl_loss, avg_reg_loss


def val_or_test(_model, _data_loader, _device, mode='baseline', verbose=0):
    """
    模型验证or测试函数
    """
    _model.eval()
    pred_loss = 0.0
    # cl_loss 和 reg_loss 只在训练阶段有效，此处省去，因此total_loss也无意义
    all_preds = []
    all_labels = []

    progress_bar = tqdm(_data_loader, desc="Validating or Testing...")

    # count = 0
    with torch.no_grad():
        for batch in progress_bar:
        #     if verbose and (count + 1) % 200 == 0:
        #         _now = datetime.now()
        #         print(f'{_now.strftime("%Y-%m-%d %H:%M:%S")}, {count + 1} of {len(_data_loader)}')
        #     count += 1

            # 数据准备
            stu_ids = batch['stu_id'].to(_device)                     # 学生ID
            labels = batch['label'].to(_device).float()               # 响应真实值
            if mode == 'fine-tune':
                input_ids = batch['input_ids'].to(_device)            # tokenize内容
                attention_mask = batch['attention_mask'].to(_device)  # 对应的mask
                # 组装为bert模型输入格式
                exer_in = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                exer_in = batch['exer_id'].to(_device)                # 题目ID

            # 前向传播
            output = _model(stu_ids, exer_in)
            _, loss_bce, _, _ = _model.get_loss(output, labels)

            # 记录结果
            pred_loss += loss_bce.item()
            preds = output[0].detach().cpu().numpy()                  # 获取预测概率
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())                   # 获取真实值

    # 计算指标
    avg_pred_loss = pred_loss / len(_data_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 二值化预测结果
    binary_preds = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_pred_loss, acc, auc, all_preds, all_labels


def my_gridsearch(_args):
    """
    自定义点搜索函数
    """
    # 定义参数网格
    param_grid = _args.param_grid
    # 生成参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # 声明最佳组合结果存储变量
    best_metrics = {'val_loss': float('inf')}
    best_params = {}

    # 遍历所有参数组合
    for i, params in enumerate(param_combinations):
        print(f"\n=== 正在训练参数组合 {i + 1}/{len(param_combinations)} ===")
        print("当前参数:", json.dumps(params, indent=2))

        # 修改待调参的参数
        _args.tau = params['tau']
        _args.lambda_reg = params['lambda_reg']
        _args.lambda_cl = params['lambda_cl']

        # 为当前参数组合创建独立目录
        param_hash = hash(frozenset(params.items()))
        _args.checkpoint_dir = f"../checkpoints/{_args.proj_name}/grid_{param_hash}"
        Path(_args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # 运行训练流程
        current_metrics = main(_args)

        # 更新最佳结果
        if current_metrics['val_loss'] < best_metrics['val_loss']:
            best_metrics = current_metrics
            best_params = params.copy()

    # 输出最终结果
    print("\n=== 网格搜索完成 ===")
    print(f"最佳参数组合: {json.dumps(best_params, indent=2)}")
    print(f"对应验证指标: loss={best_metrics['val_loss']:.4f}, acc={best_metrics['val_acc']:.4f}, auc={best_metrics['val_auc']:.4f}")
    return best_params, best_metrics


def main(args):
    """
    主函数
    """
    # 自动设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据路径配置
    data_root = f'../data/{args.data}/{args.scenario}'
    train_path = f'{data_root}/train.json'
    val_path = f'{data_root}/val.json'
    test_path = f'{data_root}/test.json'
    if 'roberta' in args.bert_path.lower():
        exer_embeds_path = f'{data_root}/exer_embeds_roberta.npy'
        exer_tokens_path = f'{data_root}/exer_tokens_roberta.json'
    elif 'bge' in args.bert_path.lower():
        exer_embeds_path = f'{data_root}/exer_embeds_bge.npy'
        exer_tokens_path = f'{data_root}/exer_tokens_bge.json'
    elif 'bert' in args.bert_path.lower():
        exer_embeds_path = f'{data_root}/exer_embeds_bert.npy'
        exer_tokens_path = f'{data_root}/exer_tokens_bert.json'
    else:
        raise ValueError(f"不支持的模型类型: {args.bert_path}，请选择 'BERT', 'RoBERTa' 或 'BGE'")

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

    # 加载模型
    if args.mode == 'baseline':
        dict_token = None  # 影响dataloader的具体形式

        # # BERT-IRT or BGE-IRT
        # model = Baseline_IRT(num_students=student_n, emb_path=exer_embeds_path).to(device)

        # BERT-MLP or BGE-MLP
        model = Baseline_MLP(num_students=student_n, emb_path=exer_embeds_path).to(device)

        # # IRT
        # model = IRT(student_n, exer_n).to(device)

        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    elif args.mode == 'freeze':
        dict_token = None  # 同上，影响dataloader的具体形式

        model = MyCDM_MLP(num_students=student_n,
                          bert_model_name=args.bert_path,
                          lora_rank=8,
                          freeze=True,
                          tau=args.tau,
                          lambda_reg=args.lambda_reg,
                          lambda_cl=args.lambda_cl,
                          emb_path=exer_embeds_path
                          ).to(device)
        
        # model = MyCDM_IRT(num_students=student_n,
        #                   bert_model_name=args.bert_path,
        #                   lora_rank=8,
        #                   freeze=True,
        #                   tau=args.tau,
        #                   lambda_reg=args.lambda_reg,
        #                   lambda_cl=args.lambda_cl,
        #                   emb_path=exer_embeds_path
        #                   ).to(device)

        # model = MyCDM_MSA(num_students=student_n,
        #                   bert_model_name=args.bert_path,
        #                   lora_rank=8,
        #                   freeze=True,
        #                   tau=args.tau,
        #                   lambda_reg=args.lambda_reg,
        #                   lambda_cl=args.lambda_cl,
        #                   emb_path=exer_embeds_path
        #                   ).to(device)

        model = Baseline_IRT_FFT(num_students=student_n,
                            # bert_model_name=args.bert_path,
                            emb_path=exer_embeds_path,
                            tau=args.tau,
                            lambda_reg=args.lambda_reg,
                            lambda_cl=args.lambda_cl,
                            ).to(device)

        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    else:  # 'fine-tune'
        dict_token = exer_tokens_path  # 同上，影响dataloader的具体形式

        # model = MyCDM_MLP_FFT(num_students=student_n,
        #                       bert_model_name=args.bert_path,
        #                       tau=args.tau,
        #                       lambda_reg=args.lambda_reg,
        #                       lambda_cl=args.lambda_cl,
        #                       ).to(device)

        # model = Basiline_MLP_FFT(num_students=student_n,
        #                          bert_model_name=args.bert_path,
        #                          tau=args.tau,
        #                          lambda_reg=args.lambda_reg,
        #                          lambda_cl=args.lambda_cl,
        #                          ).to(device)
        
        # model = Baseline_FFT(num_students=student_n,
        #                      bert_model_name=args.bert_path,
        #                      tau=args.tau,
        #                      lambda_reg=args.lambda_reg,
        #                      lambda_cl=args.lambda_cl,
        #                      ).to(device)
        
        model = Baseline_IRT_FFT(num_students=student_n,
                                 bert_model_name=args.bert_path,
                                 tau=args.tau,
                                 lambda_reg=args.lambda_reg,
                                 lambda_cl=args.lambda_cl,
                                 ).to(device)

        # 设置优化器（全量微调bert，设置分段学习率）—— 此时命令行传入的lr参数无效！
        bert_params = list(model.bert.parameters())
        # 使用参数ID进行比较
        other_params = [p for p in model.parameters() if id(p) not in [id(bp) for bp in bert_params]]
        optimizer = torch.optim.Adam(
            [
                {"params": bert_params, "lr": 2e-5},    # BERT使用较小学习率
                {"params": other_params, "lr": 1e-3}    # 其他部分使用较大学习率
            ]
        )

    # 设置Dataloader
    train_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=dict_token,  # None or path( of json)
        data_set=train_path,
        offset=0,  # 使用原始ID的数据集
        pid_zero=0
    )
    val_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=dict_token,
        data_set=val_path,
        offset=0,
        pid_zero=0
    )
    test_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=dict_token,
        data_set=test_path,
        offset=0,
        pid_zero=0
    )

    # 断点续训检查
    if os.path.exists(last_checkpoint_path):
        # 加载模型和优化器
        checkpoint = torch.load(last_checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # 覆盖相应的训练状态记录参数，打印
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        early_stop_counter = checkpoint['early_stop_counter']
        print(f"加载检查点：从epoch {start_epoch}恢复训练，当前最佳val_loss={best_val_loss:.4f}")

    # 初始化wandb
    wandb.init(
        project=args.proj_name,
        config={**vars(args)},
        resume=True if start_epoch > 0 else False,
        reinit=True if args.grid_search else False  # 是否允许重复初始化
    )

    # 训练循环
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch + 1}/{args.epoch}:")

        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"), f', training epoch {epoch + 1}')
        train_total_loss, train_pred_loss, train_cl_loss, train_reg_loss = train(
            model, train_loader, optimizer, device, mode=args.mode, verbose=args.verbose)
        print(
            f"  Train Pred Loss: {train_pred_loss:.4f}, total Loss: {train_total_loss:.4f}, CL Loss: {train_cl_loss:.4f}, Reg Loss: {train_reg_loss:.4f} ")

        now = datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, validating epoch {epoch + 1}')
        val_pred_loss, val_acc, val_auc, _, _ = val_or_test(model, val_loader, device, mode=args.mode, verbose=args.verbose)
        print(f"  Val Pred Loss: {val_pred_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        now = datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, testing epoch {epoch + 1}')
        test_pred_loss, test_acc, test_auc, _, _ = val_or_test(model, test_loader, device, mode=args.mode, verbose=args.verbose)
        print(f"  Test Pred Loss: {test_pred_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")

        # 早停逻辑（改为AUC优先）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_val_loss = val_pred_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_auc': best_val_auc
            }, best_model_path)
            print(f"发现新最佳模型，val_auc={best_val_auc:.4f}，已保存至{best_model_path}")
        else:
            early_stop_counter += 1

        # 保存最新检查点（用于断点续训）
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_auc': best_val_auc,
            'early_stop_counter': early_stop_counter
        }, last_checkpoint_path)

        # 记录训练和验证指标
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

        # 早停判断
        if early_stop_counter >= args.early_stop_patience:
            print(f"\n早停触发！连续{args.early_stop_patience}个epoch验证集无改进")
            break

    # 最终测试（加载最佳模型）
    if os.path.exists(best_model_path):
        print("\n加载最佳模型进行测试...")
        model.load_state_dict(torch.load(best_model_path)['model_state'])

    print('========================================================================')
    now = datetime.now()
    print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, testing...')
    test_pred_loss, test_acc, test_auc, y_pred, y_true = val_or_test(model, test_loader, device, mode=args.mode,
                                                                     verbose=args.verbose)
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

    # 返回验证集指标用于网格搜索比较
    return {
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_auc': best_val_auc
    }



if __name__ == '__main__':

    args_in = parse_args()

    # 格点搜索调参or直接训练
    if args_in.grid_search:
        # 指定调参字典
        args_in.param_grid = {
            'tau': [0.1],
            'lambda_reg': [1.0],
            'lambda_cl': [6.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        }
        # 调用gridsearch函数
        my_gridsearch(args_in)
    else:
        main(args_in)
