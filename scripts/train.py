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
from utils.load_data import MyDataloader
from models.model import Baseline_MLP, MyCDM_MLP


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='模型训练参数配置')

    # 实验配置
    parser.add_argument('--mode', choices=['baseline', 'freeze', 'fine-tune'], default='freeze', help='实验模式')
    parser.add_argument('--proj_name', type=str, default='freeze_250218_00', help='项目名称，用于保存检查点')
    parser.add_argument('--data', type=str, default='NIPS34', choices=['NIPS34'], help='使用的数据集名称')
    parser.add_argument('--scenario', type=str, default='all', choices=['all'], help='情景')

    # 训练超参数
    parser.add_argument('--bs', type=int, default=512, help='批次大小')
    parser.add_argument('--epoch', type=int, default=100, help='最大训练轮数')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='学习率')

    # 模型配置
    parser.add_argument('--bert_path', type=str, help='BERT预训练模型路径',
                        default='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/bert-base-uncased')

    # 训练控制
    parser.add_argument('-esp', '--early_stop_patience', type=int, default=10, help='早停等待轮数')
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

    count = 0
    for batch in _train_loader:
        if verbose and (count + 1) % 200 == 0:  # verbose=0时简化可视化输出
            _now = datetime.now()
            print(f'{_now.strftime("%Y-%m-%d %H:%M:%S")}, {count+1} of {len(_train_loader)}')
        count += 1

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

    count = 0
    with torch.no_grad():
        for batch in _data_loader:
            if verbose and (count + 1) % 200 == 0:
                _now = datetime.now()
                print(f'{_now.strftime("%Y-%m-%d %H:%M:%S")}, {count + 1} of {len(_data_loader)}')
            count += 1

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


if __name__ == '__main__':
    # 解析参数
    args = parse_args()

    # 自动设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据路径配置
    data_root = f'../data/{args.data}/{args.scenario}'
    train_path = f'{data_root}/train.json'
    val_path = f'{data_root}/val.json'
    test_path = f'{data_root}/test.json'
    exer_embeds_path = f'{data_root}/exer_embeds.npy'
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
    early_stop_counter = 0
    start_epoch = 0

    # 加载模型
    if args.mode == 'baseline':
        dict_token = None              # 影响dataloader的具体形式
        model = Baseline_MLP(num_students=student_n, emb_path=exer_embeds_path).to(device)
    elif args.mode == 'freeze':
        dict_token = None              # 同上，影响dataloader的具体形式
        model = MyCDM_MLP(num_students=student_n,
                          bert_model_name=args.bert_path,
                          lora_rank=8,
                          freeze=True,
                          tau=0.1,
                          lambda_reg=0.1,
                          lambda_cl=0.2,
                          emb_path=exer_embeds_path
                          ).to(device)
    else:  # 'fine-tune'
        dict_token = exer_tokens_path  # 同上，影响dataloader的具体形式
        model = MyCDM_MLP(num_students=student_n,
                          bert_model_name=args.bert_path,
                          lora_rank=8,
                          freeze=False,
                          tau=0.1,
                          lambda_reg=0.1,
                          lambda_cl=0.1,
                          emb_path=exer_embeds_path
                          ).to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 设置Dataloader
    train_loader = MyDataloader(
        batch_size=args.bs,
        id_to_token=dict_token,  # None or path( of json)
        data_set=train_path,
        offset=0,                # 使用原始ID的数据集
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
        config={
            "mode": args.mode,
            "batch_size": args.bs,
            "epochs": args.epoch,
            "learning_rate": args.learning_rate,
            "dataset": args.data,
            "scenario": args.scenario,
            # "model_type": ,
            "early_stop_patience": args.early_stop_patience
        },
        resume=True if start_epoch > 0 else False
    )

    # 训练循环
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch + 1}/{args.epoch}:")

        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"), f', training epoch {epoch + 1}')
        train_total_loss, train_pred_loss, train_cl_loss, train_reg_loss = train(
            model, train_loader, optimizer, device, mode=args.mode, verbose=args.verbose)
        print(f"  Train Pred Loss: {train_pred_loss:.4f}, total Loss: {train_total_loss:.4f}, CL Loss: {train_cl_loss:.4f}, Reg Loss: {train_reg_loss:.4f} ")

        now = datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")}, validating epoch {epoch + 1}')
        val_pred_loss, val_acc, val_auc, _, _ = val_or_test(model, val_loader, device, mode=args.mode, verbose=args.verbose)
        print(f"  Val Pred Loss: {val_pred_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        # 早停逻辑
        if val_pred_loss < best_val_loss:
            best_val_loss = val_pred_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"发现新最佳模型，val_loss={val_pred_loss:.4f}，已保存至{best_model_path}")
        else:
            early_stop_counter += 1

        # 保存最新检查点（用于断点续训）
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
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
    test_pred_loss, test_acc, test_auc, y_pred, y_true = val_or_test(model, test_loader, device, mode=args.mode, verbose=args.verbose)
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
