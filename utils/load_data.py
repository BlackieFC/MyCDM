import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    """
    此函数的collate功能已内化到数据处理流程中，同时集成了多种需求的输出格式
    """
    def __init__(self, data_entries, id_to_tokenized=None, offset=0, pid_zero=0):
        """
        Args:
            data_entries (list of dict): JSON格式的用户练习数据列表
            id_to_tokenized (None/dict/numpy.ndarray): 题目ID到文本分词结果（或者BERT嵌入张量）的映射字典，为None时简化为ID-Based
            offset (int): 数据中ID的偏置
            pid_zero (int): 数据中题目索引0的移动位置
        """
        self.data = data_entries
        self.id_to_tokenized = id_to_tokenized
        self.offset = offset
        self.pid_zero = pid_zero

        # 预转换所有题目特征为Tensor
        if self.id_to_tokenized is not None:
            self._preprocess_tokenized()

    def _preprocess_tokenized(self):
        """
        将分词结果预先转换为Tensor加速后续处理
        """
        # if isinstance(self.id_to_tokenized, np.ndarray):  # embedding-based 方法效率低下，暂不使用
        #     # (BERT-)embedding-based
        #     self.id_to_tokenized = torch.tensor(self.id_to_tokenized)  # 保持BERT嵌入的默认数据格式
        # else:
        # (dict) token-based
        for key in ['input_ids', 'attention_mask']:
            self.id_to_tokenized[key] = [
                torch.tensor(x, dtype=torch.int64)
                for x in self.id_to_tokenized[key]
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        user_id = torch.tensor(entry['user_id'], dtype=torch.long)
        exer_id = torch.tensor(entry['exer_id'], dtype=torch.long)
        score = torch.tensor(entry['score'], dtype=torch.float)
        if self.offset != 0:
            # 将ID转换回原始ID
            user_id = user_id - self.offset
            exer_id = exer_id - self.offset
        if self.pid_zero != 0:
            # 将特定ID转换回0
            exer_id[exer_id == self.pid_zero] = 0

        if isinstance(self.id_to_tokenized, dict):
            # token-based
            return {
                'stu_id': user_id,
                'label': score,
                'input_ids': self.id_to_tokenized['input_ids'][exer_id],
                'attention_mask': self.id_to_tokenized['attention_mask'][exer_id]
            }
        # elif isinstance(self.id_to_tokenized, torch.Tensor):  # embedding-based 方法效率低下，暂不使用
        #     # embedding-based
        #     return {
        #         'stu_id': user_id,
        #         'label': score,
        #         'exer_emb': self.id_to_tokenized[exer_id]
        #     }
        else:
            # ID-based
            return {
                'stu_id': user_id,
                'label': score,
                'exer_id': exer_id
            }


def MyDataloader(data_set,
                 batch_size=2,
                 id_to_token=None,
                 offset=0,
                 pid_zero=0
                 ):
    """
    调用 MyDataset 和 torch.utils.data.DataLoader，获取 Dataloader封装
    :param data_set     : (MyDataset参数, list of dict) JSON格式的用户练习数据列表
    :param batch_size   :
    :param id_to_token  : (MyDataset参数, None/dict/numpy.ndarray) 题目ID到文本分词结果（或者BERT嵌入张量）的映射字典，为None时简化为ID-Based
    :param offset       : (MyDataset参数) 数据中ID的偏置
    :param pid_zero     : (MyDataset参数) 数据中题目索引0的移动位置
    :return:
    """
    # 读取数据集
    with open(data_set, 'r', encoding='utf-8') as fi:
        data_set = json.load(fi)
    # 读取题目文本分词or嵌入数据（如有）
    if id_to_token is not None and id_to_token.endswith('.json'):
        with open(id_to_token, 'r', encoding='utf-8') as fi:
            id_to_token = json.load(fi)     # json dict
    elif id_to_token is not None:
        id_to_token = np.load(id_to_token)  # numpy.ndarray

    # 封装Dataset
    _dataset = MyDataset(data_set, id_to_token, offset=offset, pid_zero=pid_zero)
    # 封装Dataloader
    _dataloader = DataLoader(_dataset, batch_size=batch_size)

    return _dataloader
