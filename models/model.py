import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaModel
from peft import LoraConfig, get_peft_model
import numpy as np
from utils.loss import custom_loss, cl_and_reg_loss


class ConstrainedEmbedding(nn.Module):
    """
    初始化nn.Embedding层时，要求嵌入结果的L2范数等于15，并在后续训练过程中持续对嵌入层权重进行约束，使得嵌入结果的L2范数始终不变
    """
    def __init__(self, num_embeddings, embedding_dim, norm=15.0):
        super().__init__()
        self.norm = norm
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 初始化参数
        self._initialize_weights()
        # 注册前向钩子保持约束
        self.embedding.register_forward_hook(self._apply_constraint)

    def _initialize_weights(self):
        """初始化使每个嵌入向量L2范数=self.norm"""
        with torch.no_grad():
            # 先进行标准初始化
            nn.init.normal_(self.embedding.weight, mean=0, std=1)
            # 计算并规范化范数
            norms = torch.norm(self.embedding.weight, p=2, dim=1, keepdim=True)
            self.embedding.weight.data = self.norm * (self.embedding.weight.data / norms)

    def _apply_constraint(self, module, input, output):
        """前向传播时自动约束权重"""
        with torch.no_grad():
            # 计算当前范数
            norms = torch.norm(module.weight, p=2, dim=1, keepdim=True)
            # 重新缩放权重
            module.weight.data = self.norm * (module.weight.data / norms)

        return output

    def forward(self, x):
        return self.embedding(x)


class Baseline_MLP(nn.Module):
    """
    MLP-Baseline模型函数：
        （1）题目输入：ID输入——nn.Embedding权重固定为BERT嵌入
        （2）学生表征：普通单嵌入
        （3）端到端预测头（MLP）
    """
    def __init__(self, num_students,
                 emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        self.emb_path = emb_path
        self.bert = self.get_exer_embed_layer()    # nn.Embedding(948, 768)，固定的BERT嵌入
        self.d_model = self.bert.weight.shape[1]

        # proficiency：普通单嵌入
        self.stu_emb = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )

        # MLP预测头
        self.prednet = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model,),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_emb.weight, mean=0.0, std=0.1)
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            exer_in: 题目ID张量[batch_size]
        """
        # 学生嵌入
        proficiency = self.stu_emb(stu_ids)  # [batch_size, 768]
        # 题目嵌入
        exer_emb = self.bert(exer_in)        # [batch_size, 768]
        # prednet
        logits = self.prednet(torch.cat([exer_emb, proficiency], dim=1))
        output = torch.sigmoid(logits).squeeze(-1)

        return output, exer_emb, proficiency

    @staticmethod
    def get_loss(output, labels):
        """计算总损失"""
        preds, _, _ = output
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size], BCE损失
        cl_loss = torch.zeros_like(bce_loss, requires_grad=False)
        reg_loss = torch.zeros_like(bce_loss, requires_grad=False)
        total_loss = bce_loss + cl_loss + reg_loss
        return total_loss, bce_loss, cl_loss, reg_loss


class Baseline_IRT(nn.Module):
    """
    IRT-Baseline模型函数：
        （1）题目输入：ID输入——nn.Embedding权重固定为BERT嵌入
        （2）学生表征：普通单嵌入
        （3）类IRT预测头
    """
    def __init__(self, num_students,
                 emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        self.emb_path = emb_path
        self.bert = self.get_exer_embed_layer()    # nn.Embedding(948, 768)，固定的BERT嵌入
        self.d_model = self.bert.weight.shape[1]   # IRT-Baseline中为映射层的输入维度

        # proficiency：普通单嵌入
        self.stu_emb = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=1                        # IRT-Baseline，特征维度为 1
        )

        # 题目难度和区分度映射层
        self.proj_disc = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.Sigmoid(),                          # 引入非线性
            nn.Linear(2 * self.d_model, 1)
        )
        self.proj_diff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),                          # 引入非线性
            nn.Linear(self.d_model, 1)
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_emb.weight, mean=0.0, std=0.1)
        for module in self.proj_diff:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.proj_disc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            exer_in: 题目ID张量 [batch_size]
        """
        # 学生嵌入
        proficiency = self.stu_emb(stu_ids)          # [batch_size, 1]
        # 题目嵌入
        exer_emb = self.bert(exer_in)                # [batch_size, 768]
        # 类IRT预测头
        a = torch.sigmoid(self.proj_disc(exer_emb))  # [batch_size, 1]
        b = self.proj_diff(exer_emb)                 # [batch_size, 1]
        output = 1 / (1 + torch.exp(-1.703 * a * (proficiency - b)))
        output = output.squeeze(-1)                  # [batch_size, 1] -> [batch_size]

        return output, exer_emb, proficiency

    @staticmethod
    def get_loss(output, labels):
        """计算总损失"""
        preds, _, _ = output
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size], BCE损失
        cl_loss = torch.zeros_like(bce_loss, requires_grad=False)
        reg_loss = torch.zeros_like(bce_loss, requires_grad=False)
        total_loss = bce_loss + cl_loss + reg_loss
        return total_loss, bce_loss, cl_loss, reg_loss


class MyCDM_MLP(nn.Module):
    """模型函数"""
    def __init__(self, num_students, bert_model_name='bert-base-uncased', lora_rank=8, freeze=True, tau=0.1, lambda_reg=1.0,
                 lambda_cl=10.0, emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        if bert_model_name is None:
            self.bert = nn.Embedding(num_embeddings=948, embedding_dim=768)  # 临时
            nn.init.normal_(self.bert.weight, mean=0.0, std=0.15)
        elif freeze:
            self.emb_path = emb_path
            self.bert = self.get_exer_embed_layer()  # nn.Embedding(948, 768)，固定的BERT嵌入
            """若冻结BERT权重，则不带着模型训练，效率太低——对应需求题目ID形式的输入"""
            # # 读取预训练BERT
            # self.bert = BertModel.from_pretrained(bert_model_name)
            # # 冻结BERT参数
            # for param in self.bert.parameters():
            #     param.requires_grad = False
        else:
            """若使用LoRA对BERT进行微调，则需要把题目文本组织为字典形式的分词结果作为BERT输入"""
            # 读取预训练BERT
            self.bert = BertModel.from_pretrained(bert_model_name)
            # 配置LoRA适配器
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            # 应用LoRA适配器
            self.bert = get_peft_model(self.bert, lora_config)
            # 冻结BERT主干参数
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.d_model = self.bert.weight.shape[1]
        self.freeze = freeze
        self.bert_model_name = bert_model_name

        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(  # ConstrainedEmbedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model        # norm=15.0
        )
        self.stu_neg = nn.Embedding(  # ConstrainedEmbedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model        # norm=15.0
        )

        # MLP预测头
        # self.prednet = nn.Sequential(
        #     nn.Linear(2 * self.d_model, 2 * self.d_model),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2 * self.d_model, self.d_model),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.d_model, 1)
        # )
        self.prednet = nn.Sequential(
            nn.Linear(3 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.15)
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.15)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            （1）exer_in: 包含文本输入的字典
                    input_ids       : [batch_size, seq_len]
                    attention_mask  : [batch_size, seq_len]
            （2）exer_in: 题目ID张量 [batch_size]
        """
        # 学生的正负双模态表征
        u_pos = self.stu_pos(stu_ids)      # [batch_size, 768]
        u_neg = self.stu_neg(stu_ids)      # [batch_size, 768]
        # 题目表征
        if self.freeze or self.bert_model_name is None:
            exer_emb = self.bert(exer_in)  # [batch_size, 768]
        else:
            bert_output = self.bert(       # [batch_size, 768]，提取CLS token作为题目嵌入
                input_ids=exer_in["input_ids"],
                attention_mask=exer_in["attention_mask"])
            exer_emb = bert_output.last_hidden_state[:, 0, :]

        # u_pos = u_pos / u_pos.norm(dim=-1, keepdim=True)
        # u_neg = u_neg / u_neg.norm(dim=-1, keepdim=True)
        # exer_emb = exer_emb / exer_emb.norm(dim=-1, keepdim=True)

        # MLP预测头
        # stu_emb = u_pos - u_neg            # [batch_size, 768]
        # logits = self.prednet(torch.cat([exer_emb, stu_emb], dim=1))  # [batch_size, 1]
        logits = self.prednet(torch.cat([exer_emb, torch.multiply(exer_emb, u_pos), torch.multiply(exer_emb, u_neg)], dim=1))  # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)  # [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        loss_contrast, loss_reg = custom_loss(exer_emb, u_pos, u_neg, labels, self.tau, norm=False, delta=None)
        # loss_contrast, loss_reg = cl_and_reg_loss(exer_emb, u_pos, u_neg, labels, self.tau, delta=0.1, norm=True)
        # 总损失
        if self.lambda_cl < 1:
            total_loss = (1-self.lambda_cl)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        else:
            total_loss = bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class MyCDM_MLP_backup(nn.Module):
    """模型函数"""
    def __init__(self, num_students, bert_model_name='bert-base-uncased', lora_rank=8, freeze=True, tau=0.1, lambda_reg=1.0,
                 lambda_cl=10.0, emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        if bert_model_name is None:
            self.bert = nn.Embedding(num_embeddings=948, embedding_dim=768)  # 临时
            nn.init.normal_(self.bert.weight, mean=0.0, std=0.1)
        elif freeze:
            self.emb_path = emb_path
            self.bert = self.get_exer_embed_layer()  # nn.Embedding(948, 768)，固定的BERT嵌入
            """若冻结BERT权重，则不带着模型训练，效率太低——对应需求题目ID形式的输入"""
            # # 读取预训练BERT
            # self.bert = BertModel.from_pretrained(bert_model_name)
            # # 冻结BERT参数
            # for param in self.bert.parameters():
            #     param.requires_grad = False
        else:
            """若使用LoRA对BERT进行微调，则需要把题目文本组织为字典形式的分词结果作为BERT输入"""
            # 读取预训练BERT
            self.bert = BertModel.from_pretrained(bert_model_name)
            # 配置LoRA适配器
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            # 应用LoRA适配器
            self.bert = get_peft_model(self.bert, lora_config)
            # 冻结BERT主干参数
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.d_model = self.bert.weight.shape[1]
        self.freeze = freeze
        self.bert_model_name = bert_model_name

        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(  # ConstrainedEmbedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model        # norm=15.0
        )
        self.stu_neg = nn.Embedding(  # ConstrainedEmbedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model        # norm=15.0
        )

        # MLP预测头
        # self.prednet = nn.Sequential(
        #     nn.Linear(2 * self.d_model, 2 * self.d_model),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2 * self.d_model, self.d_model),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.d_model, 1)
        # )
        self.prednet = nn.Sequential(
            nn.Linear(3 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.15)
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.15)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            （1）exer_in: 包含文本输入的字典
                    input_ids       : [batch_size, seq_len]
                    attention_mask  : [batch_size, seq_len]
            （2）exer_in: 题目ID张量 [batch_size]
        """
        # 学生的正负双模态表征
        u_pos = self.stu_pos(stu_ids)      # [batch_size, 768]
        u_neg = self.stu_neg(stu_ids)      # [batch_size, 768]
        # 题目表征
        if self.freeze or self.bert_model_name is None:
            exer_emb = self.bert(exer_in)  # [batch_size, 768]
        else:
            bert_output = self.bert(       # [batch_size, 768]，提取CLS token作为题目嵌入
                input_ids=exer_in["input_ids"],
                attention_mask=exer_in["attention_mask"])
            exer_emb = bert_output.last_hidden_state[:, 0, :]

        # u_pos = u_pos / u_pos.norm(dim=-1, keepdim=True)
        # u_neg = u_neg / u_neg.norm(dim=-1, keepdim=True)
        # exer_emb = exer_emb / exer_emb.norm(dim=-1, keepdim=True)

        # MLP预测头
        # stu_emb = u_pos - u_neg            # [batch_size, 768]
        # logits = self.prednet(torch.cat([exer_emb, stu_emb], dim=1))  # [batch_size, 1]
        logits = self.prednet(torch.cat([exer_emb, u_pos, u_neg], dim=1))  # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)  # [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        # loss_contrast, loss_reg = custom_loss(exer_emb, u_pos, u_neg, labels, self.tau, norm=True, delta=0.1)
        loss_contrast, loss_reg = cl_and_reg_loss(exer_emb, u_pos, u_neg, labels, self.tau, delta=0.1, norm=True)
        # 总损失
        if self.lambda_cl < 1:
            total_loss = (1-self.lambda_cl)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        else:
            total_loss = bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class MyCDM_IRT(nn.Module):
    """模型函数"""

    def __init__(self, num_students, bert_model_name='bert-base-uncased', lora_rank=8, freeze=True, tau=0.1, lambda_reg=0.1,
                 lambda_cl=0.1, emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        if freeze:
            self.emb_path = emb_path
            self.bert = self.get_exer_embed_layer()  # nn.Embedding(948, 768)，固定的BERT嵌入
            """若冻结BERT权重，则不带着模型训练，效率太低——对应需求题目ID形式的输入"""
            # # 读取预训练BERT
            # self.bert = BertModel.from_pretrained(bert_model_name)
            # # 冻结BERT参数
            # for param in self.bert.parameters():
            #     param.requires_grad = False
        else:
            """若使用LoRA对BERT进行微调，则需要把题目文本组织为字典形式的分词结果作为BERT输入"""
            # 读取预训练BERT
            self.bert = BertModel.from_pretrained(bert_model_name)
            # 配置LoRA适配器
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            # 应用LoRA适配器
            self.bert = get_peft_model(self.bert, lora_config)
            # 冻结BERT主干参数
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.d_model = self.bert.weight.shape[1]  # IRT预测头时为映射层的输入维度
        self.freeze = freeze

        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model            # 对应于IRT预测头
        )
        self.stu_neg = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )

        # 题目难度和区分度映射层
        self.proj_disc = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.Sigmoid(),  # 引入非线性
            nn.Linear(2 * self.d_model, 1)
        )
        self.proj_diff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),  # 引入非线性
            nn.Linear(self.d_model, 1),
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.15)
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.15)
        for module in self.proj_diff:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.proj_disc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            （1）exer_in: 包含文本输入的字典
                    input_ids       : [batch_size, seq_len]
                    attention_mask  : [batch_size, seq_len]
            （2）exer_in: 题目ID张量 [batch_size]
        """
        # 正负学生表征
        u_pos = self.stu_pos(stu_ids)                # [batch_size, 768]
        u_neg = self.stu_neg(stu_ids)                # [batch_size, 768]
        # 题目表征
        if self.freeze:
            exer_emb = self.bert(exer_in)            # [batch_size, 1]
        else:
            bert_output = self.bert(                 # [batch_size, 1]，提取CLS token作为题目嵌入
                input_ids=exer_in["input_ids"],
                attention_mask=exer_in["attention_mask"])
            exer_emb = bert_output.last_hidden_state[:, 0, :]

        # 类IRT预测头
        a = torch.sigmoid(self.proj_disc(exer_emb))  # [batch_size, 1]
        b = self.proj_diff(exer_emb)                 # [batch_size, 1]
        output = 1 / (1 + torch.exp(-1.703 * a * (torch.sum(torch.multiply(exer_emb, u_pos-u_neg), dim=1, keepdim=True) - b)))
        output = output.squeeze(-1)  # [batch_size, 1] -> [batch_size]

        # # 类MIRT预测头
        # output = 1 / (1 + torch.exp(-1.703 * (torch.sum(torch.multiply(a, u_pos-u_neg), dim=-1, keepdim=True) - b)))
        # output = output.squeeze(-1)  # [batch_size, n_know] -> [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        # loss_contrast, loss_reg = custom_loss(exer_emb, u_pos, u_neg, labels, self.tau, norm=True, delta=0.1)
        loss_contrast, loss_reg = cl_and_reg_loss(exer_emb, u_pos, u_neg, labels, self.tau, delta=0.1, norm=True)
        # 总损失
        if self.lambda_cl < 1:
            total_loss = (1-self.lambda_cl)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        else:
            total_loss = bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class MyCDM_IRT_backup(nn.Module):
    """模型函数"""

    def __init__(self, num_students, bert_model_name='bert-base-uncased', lora_rank=8, freeze=True, tau=0.1, lambda_reg=0.1,
                 lambda_cl=0.1, emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        if freeze:
            self.emb_path = emb_path
            self.bert = self.get_exer_embed_layer()  # nn.Embedding(948, 768)，固定的BERT嵌入
            """若冻结BERT权重，则不带着模型训练，效率太低——对应需求题目ID形式的输入"""
            # # 读取预训练BERT
            # self.bert = BertModel.from_pretrained(bert_model_name)
            # # 冻结BERT参数
            # for param in self.bert.parameters():
            #     param.requires_grad = False
        else:
            """若使用LoRA对BERT进行微调，则需要把题目文本组织为字典形式的分词结果作为BERT输入"""
            # 读取预训练BERT
            self.bert = BertModel.from_pretrained(bert_model_name)
            # 配置LoRA适配器
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            # 应用LoRA适配器
            self.bert = get_peft_model(self.bert, lora_config)
            # 冻结BERT主干参数
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.d_model = self.bert.weight.shape[1]  # IRT预测头时为映射层的输入维度
        self.freeze = freeze

        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=1                       # 对应于IRT预测头
        )
        self.stu_neg = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=1
        )

        # 题目难度和区分度映射层
        self.proj_disc = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.Sigmoid(),  # 引入非线性
            nn.Linear(2 * self.d_model, 1)
        )
        self.proj_diff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),  # 引入非线性
            nn.Linear(self.d_model, 1),
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.1)
        for module in self.proj_diff:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.proj_disc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            （1）exer_in: 包含文本输入的字典
                    input_ids       : [batch_size, seq_len]
                    attention_mask  : [batch_size, seq_len]
            （2）exer_in: 题目ID张量 [batch_size]
        """
        # 正负学生表征
        u_pos = self.stu_pos(stu_ids)                # [batch_size, 1]
        u_neg = self.stu_neg(stu_ids)                # [batch_size, 1]
        # 题目表征
        if self.freeze:
            exer_emb = self.bert(exer_in)            # [batch_size, 1]
        else:
            bert_output = self.bert(                 # [batch_size, 1]，提取CLS token作为题目嵌入
                input_ids=exer_in["input_ids"],
                attention_mask=exer_in["attention_mask"])
            exer_emb = bert_output.last_hidden_state[:, 0, :]

        # 类IRT预测头
        a = torch.sigmoid(self.proj_disc(exer_emb))  # [batch_size, 1]
        b = self.proj_diff(exer_emb)                 # [batch_size, 1]
        output = 1 / (1 + torch.exp(-1.703 * a * (u_pos - u_neg - b)))
        output = output.squeeze(-1)  # [batch_size, 1] -> [batch_size]

        # # 类MIRT预测头
        # output = 1 / (1 + torch.exp(-1.703 * (torch.sum(torch.multiply(a, u_pos-u_neg), dim=-1, keepdim=True) - b)))
        # output = output.squeeze(-1)  # [batch_size, n_know] -> [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        loss_contrast, loss_reg = custom_loss(exer_emb, u_pos, u_neg, labels, self.tau)
        # 总损失
        total_loss = (1-self.lambda_cl-self.lambda_reg)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class IRT(nn.Module):
    """
    IRT
    """
    def __init__(self, student_n, exer_n, ratio=1.703, a_range=1, sigmoid=False):
        """
        :param student_n: 学生数
        :param exer_n   : 题目数
        :param d_model  : 嵌入特征维度
        :param ratio    : 指数倍率因子
        :param exer_emb : 题目嵌入张量
        :param n_choice : 题目选项数
        :param a_range  : 区分度取值范围
        """
        self.student_n = student_n
        self.exer_n = exer_n
        self.ratio = ratio
        self.a_range = a_range
        self.sigmoid = sigmoid

        super(IRT, self).__init__()

        # 嵌入层
        self.student_emb = nn.Embedding(self.student_n, 1)
        self.proj_disc = nn.Embedding(self.exer_n, 1)    # exer_id -> a
        self.proj_diff = nn.Embedding(self.exer_n, 1)    # exer_id -> b

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id):
        """
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :return: FloatTensor, the probabilities of answering correctly
        """
        # 映射为具体项
        theta = self.student_emb(stu_id)
        a = torch.sigmoid(self.proj_disc(exer_id)) * self.a_range
        b = self.proj_diff(exer_id)
        if self.sigmoid:
            theta = torch.sigmoid(theta)
            b = torch.sigmoid(b)
        # 交互函数
        output = 1 / (1 + torch.exp(-self.ratio * a * (theta - b)))

        return output.squeeze(), b, theta
    
    @staticmethod
    def get_loss(output, labels):
        """计算总损失"""
        preds, _, _ = output
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size], BCE损失
        cl_loss = torch.zeros_like(bce_loss, requires_grad=False)
        reg_loss = torch.zeros_like(bce_loss, requires_grad=False)
        total_loss = bce_loss + cl_loss + reg_loss
        return total_loss, bce_loss, cl_loss, reg_loss


class MyCDM_MSA(nn.Module):
    """模型函数"""
    def __init__(self, num_students, bert_model_name='bert-base-uncased', lora_rank=8, freeze=True, tau=0.1, lambda_reg=1.0,
                 lambda_cl=10.0, emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy'):
        """初始化模型结构"""
        super().__init__()
        if bert_model_name is None:
            self.bert = nn.Embedding(num_embeddings=948, embedding_dim=768)  # 临时
            nn.init.normal_(self.bert.weight, mean=0.0, std=0.1)
        elif freeze:
            self.emb_path = emb_path
            self.bert = self.get_exer_embed_layer()  # nn.Embedding(948, 768)，固定的BERT嵌入
            """若冻结BERT权重，则不带着模型训练，效率太低——对应需求题目ID形式的输入"""
            # # 读取预训练BERT
            # self.bert = BertModel.from_pretrained(bert_model_name)
            # # 冻结BERT参数
            # for param in self.bert.parameters():
            #     param.requires_grad = False
        else:
            """若使用LoRA对BERT进行微调，则需要把题目文本组织为字典形式的分词结果作为BERT输入"""
            # 读取预训练BERT
            self.bert = BertModel.from_pretrained(bert_model_name)
            # 配置LoRA适配器
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            # 应用LoRA适配器
            self.bert = get_peft_model(self.bert, lora_config)
            # 冻结BERT主干参数
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.d_model = self.bert.weight.shape[1]
        self.freeze = freeze
        self.bert_model_name = bert_model_name

        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(  # ConstrainedEmbedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model        # norm=15.0
        )
        self.stu_neg = nn.Embedding(  # ConstrainedEmbedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model        # norm=15.0
        )

        # MLP预测头
        self.attn_pos = nn.MultiheadAttention(self.d_model, num_heads=8, batch_first=True)  # (bs, 1, d_model)
        self.attn_neg = nn.MultiheadAttention(self.d_model, num_heads=8, batch_first=True)
        self.prednet = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )

        # 初始化参数
        self.initialize()

    def get_exer_embed_layer(self):
        """读取题目文本嵌入"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)  # (948, 768)
        return nn.Embedding.from_pretrained(kc_embeds, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.15)
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.15)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # MHA
        for name, param in self.attn_pos.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for name, param in self.attn_neg.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            （1）exer_in: 包含文本输入的字典
                    input_ids       : [batch_size, seq_len]
                    attention_mask  : [batch_size, seq_len]
            （2）exer_in: 题目ID张量 [batch_size]
        """
        # 学生的正负双模态表征
        u_pos = self.stu_pos(stu_ids)      # [batch_size, 768]
        u_neg = self.stu_neg(stu_ids)      # [batch_size, 768]
        # 题目表征
        if self.freeze or self.bert_model_name is None:
            exer_emb = self.bert(exer_in)  # [batch_size, 768]
        else:
            bert_output = self.bert(       # [batch_size, 768]，提取CLS token作为题目嵌入
                input_ids=exer_in["input_ids"],
                attention_mask=exer_in["attention_mask"])
            exer_emb = bert_output.last_hidden_state[:, 0, :]

        u_pos = u_pos.unsqueeze(1)
        u_neg = u_neg.unsqueeze(1)
        exer_emb = exer_emb.unsqueeze(1)
        attn_pos, _ = self.attn_pos(exer_emb, u_pos, u_pos)
        attn_neg, _ = self.attn_pos(exer_emb, u_neg, u_neg)
        attn_pos = attn_pos.squeeze(1)
        attn_neg = attn_neg.squeeze(1)
        u_pos = u_pos.squeeze(1)
        u_neg = u_neg.squeeze(1)
        exer_emb = exer_emb.squeeze(1)
        logits = self.prednet(torch.cat([attn_pos, attn_neg], dim=1))  # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)  # [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        # loss_contrast, loss_reg = custom_loss(exer_emb, u_pos, u_neg, labels, self.tau, norm=True, delta=0.1)
        loss_contrast, loss_reg = cl_and_reg_loss(exer_emb, u_pos, u_neg, labels, self.tau, delta=0.1, norm=True)
        # 总损失
        if self.lambda_cl < 1:
            total_loss = (1-self.lambda_cl)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        else:
            total_loss = bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class MyCDM_MLP_kmean(nn.Module):
    """模型函数"""
    def __init__(self,
                 num_students,
                 tau=0.1,
                 lambda_reg=1.0,
                 lambda_cl=10.0,
                 emb_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_KCandExer_emb.npy',  # (948,768)
                 kmean_path='/mnt/new_pfs/liming_team/auroraX/songchentao/llama/exer_text/nips34_kmean_emb.npy',    # (32,768)
                 n_topk=3,
                 ):
        """初始化模型结构"""
        super().__init__()
        self.emb_path = emb_path
        self.clu_path = kmean_path
        self.n_topk = n_topk
        self.bert = self.get_embed_layer()  # nn.Embedding(948, 32)，基于聚类结果的嵌入层

        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.d_model = self.bert.weight.shape[1]

        # 学生能力嵌入层（u+和u-）—— 映射到聚类数，同时调用时需要后接sigmoid
        self.stu_pos = nn.Embedding(
            num_embeddings=num_students,  # 注意嵌入层+sigmoid才是非负的能力表征
            embedding_dim=self.d_model    # 嵌入层仅得到logits
        )
        self.stu_neg = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )
        # MLP预测头
        self.prednet = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )
        # 初始化参数
        self.initialize()

    def get_embed_layer(self):
        """读取题目文本嵌入（基于K-mean聚类）"""
        kc_embeds = np.load(self.emb_path)
        kc_embeds = torch.tensor(kc_embeds)         # (948, 768)
        kc_embeds = kc_embeds / kc_embeds.norm(p=2, dim=1, keepdim=True)

        km_center = np.load(self.clu_path)
        km_center = torch.tensor(km_center)         # (32, 768)
        km_center = km_center / km_center.norm(p=2, dim=1, keepdim=True)

        sim = torch.matmul(kc_embeds, km_center.T)  # (948,32)

        if self.n_topk:
            # 找到每行的top-k值及其索引
            _, top_k_idx = torch.topk(sim, k=min(self.n_topk, sim.size(1)), dim=1)
            # 创建掩码矩阵
            mask = torch.zeros_like(sim, dtype=torch.bool)
            batch_indices = torch.arange(sim.size(0)).unsqueeze(1).expand(-1, top_k_idx.size(1))
            mask[batch_indices, top_k_idx] = True
            # 使用掩码将非top-k的位置设为负无穷大
            C_masked = torch.where(mask, sim, torch.tensor(float('-inf')))
            # 应用softmax函数
            sparse_softmax_output = F.softmax(C_masked, dim=1)
        else:
            # 逐行距平
            sim = sim - torch.mean(sim, dim=1, keepdim=True)
            # 应用softmax函数
            sparse_softmax_output = F.softmax(sim, dim=1)

        return nn.Embedding.from_pretrained(sparse_softmax_output, freeze=True)

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.1)  # sigmoid后非负，保证取0-1中间值
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.1)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        """
        输入：
            stu_ids: 学生ID张量 [batch_size]
            （1）exer_in: 包含文本输入的字典
                    input_ids       : [batch_size, seq_len]
                    attention_mask  : [batch_size, seq_len]
            （2）exer_in: 题目ID张量 [batch_size]
        """
        # 学生的正负双模态表征（需要补充上sigmoid）
        u_pos = torch.sigmoid(self.stu_pos(stu_ids))      # [batch_size, 32]
        u_neg = torch.sigmoid(self.stu_neg(stu_ids))      # [batch_size, 32]
        # 题目表征
        exer_emb = self.bert(exer_in)  # [batch_size, 768]

        # MLP预测头
        # stu_emb = u_pos - u_neg            # [batch_size, 768]
        # logits = self.prednet(torch.cat([exer_emb, stu_emb], dim=1))  # [batch_size, 1]
        logits = self.prednet(torch.cat([exer_emb, u_pos, u_neg], dim=1))  # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)  # [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        # loss_contrast, loss_reg = custom_loss(exer_emb, u_pos, u_neg, labels, self.tau, norm=True, delta=0.1)
        loss_contrast, loss_reg = cl_and_reg_loss(exer_emb, u_pos, u_neg, labels, self.tau, delta=0.1, norm=True)
        # 总损失
        if self.lambda_cl < 1:
            total_loss = (1-self.lambda_cl)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        else:
            total_loss = bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class MyCDM_MLP_FFT(nn.Module):
    """
    使用全量微调BERT，需要把题目文本组织为字典形式的分词结果作为BERT输入
        # 全量微调模式下，所有BERT参数都可以训练，不需要冻结参数
        # 默认情况下，所有参数都已经是requires_grad=True的状态
    """
    def __init__(self, num_students, bert_model_name='bert-base-uncased', tau=0.1, lambda_reg=1.0, lambda_cl=0.5):
        super().__init__()
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl

        # 读取预训练BERT & 解冻所有参数
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.d_model = self.bert.config.hidden_size
        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )
        self.stu_neg = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )
        # MLP预测头
        self.prednet = nn.Sequential(
            nn.Linear(3 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )
        # 初始化参数
        self.initialize()

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.1)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        # 学生的正负双模态表征
        u_pos = self.stu_pos(stu_ids)      # [batch_size, 768]
        u_neg = self.stu_neg(stu_ids)      # [batch_size, 768]
        # 题目表征
        bert_output = self.bert(           # [batch_size, 768]，提取CLS token作为题目嵌入
            input_ids=exer_in["input_ids"],
            attention_mask=exer_in["attention_mask"])
        exer_emb = bert_output.last_hidden_state[:, 0, :]

        """MLP预测头（1）不与问题交互"""
        # stu_emb = u_pos - u_neg                                       # [batch_size, 768]
        # logits = self.prednet(torch.cat([exer_emb, stu_emb], dim=1))  # [batch_size, 1]
        """MLP预测头（2）与问题交互"""
        logits = self.prednet(torch.cat([exer_emb, torch.multiply(exer_emb, u_pos), torch.multiply(exer_emb, u_neg)], dim=1))  # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)                      # [batch_size]

        return output, exer_emb, u_pos, u_neg

    def get_loss(self, output, labels):
        """计算总损失"""
        preds, exer_emb, u_pos, u_neg = output
        # BCE损失 <=> 预测损失
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size]
        # 对比损失 & 正则化项
        loss_contrast, loss_reg = cl_and_reg_loss(exer_emb, u_pos, u_neg, labels, self.tau)
        # 总损失
        if self.lambda_cl < 1:
            total_loss = (1-self.lambda_cl)*bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        else:
            total_loss = bce_loss + self.lambda_cl*loss_contrast + self.lambda_reg*loss_reg
        return total_loss, bce_loss, loss_contrast, loss_reg


class Basiline_MLP_FFT(nn.Module):
    """
    使用全量微调BERT，需要把题目文本组织为字典形式的分词结果作为BERT输入
        # 全量微调模式下，所有BERT参数都可以训练，不需要冻结参数
        # 默认情况下，所有参数都已经是requires_grad=True的状态
    """
    def __init__(self, num_students, bert_model_name='bert-base-uncased', tau=0.1, lambda_reg=1.0, lambda_cl=0.5):
        super().__init__()
        # self.tau = tau
        # self.lambda_reg = lambda_reg
        # self.lambda_cl = lambda_cl

        # 读取预训练BERT & 解冻所有参数
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.d_model = self.bert.config.hidden_size
        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )
        # self.stu_neg = nn.Embedding(
        #     num_embeddings=num_students,
        #     embedding_dim=self.d_model
        # )
        # MLP预测头
        self.prednet = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )
        # 初始化参数
        self.initialize()

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.1)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        # 学生的正负双模态表征
        u_pos = self.stu_pos(stu_ids)      # [batch_size, 768]
        # u_neg = self.stu_neg(stu_ids)      # [batch_size, 768]
        # 题目表征
        bert_output = self.bert(           # [batch_size, 768]，提取CLS token作为题目嵌入
            input_ids=exer_in["input_ids"],
            attention_mask=exer_in["attention_mask"])
        exer_emb = bert_output.last_hidden_state[:, 0, :]

        """MLP预测头（1）不与问题交互"""
        # stu_emb = u_pos - u_neg                                       # [batch_size, 768]
        # logits = self.prednet(torch.cat([exer_emb, stu_emb], dim=1))  # [batch_size, 1]
        """MLP预测头（2）与问题交互"""
        logits = self.prednet(torch.cat([exer_emb, u_pos], dim=1))      # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)                      # [batch_size]

        return output, exer_emb, u_pos, u_pos
    
    @staticmethod
    def get_loss(output, labels):
        """计算总损失"""
        preds, _, _, _ = output
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size], BCE损失
        cl_loss = torch.zeros_like(bce_loss, requires_grad=False)
        reg_loss = torch.zeros_like(bce_loss, requires_grad=False)
        total_loss = bce_loss + cl_loss + reg_loss
        return total_loss, bce_loss, cl_loss, reg_loss


class Baseline_FFT(nn.Module):
    """
    使用全量微调基础模型，需要把题目文本组织为字典形式的分词结果作为模型输入
        # 全量微调模式下，所有模型参数都可以训练，不需要冻结参数
        # 默认情况下，所有参数都已经是requires_grad=True的状态
    """
    def __init__(self, num_students, model_name=None, tau=0.1, lambda_reg=1.0, lambda_cl=0.5):
        super().__init__()
        # self.tau = tau
        # self.lambda_reg = lambda_reg
        # self.lambda_cl = lambda_cl

        # 根据选择加载不同的模型
        if 'roberta' in model_name.lower():
            self.model = XLMRobertaModel.from_pretrained(model_name or '/mnt/new_pfs/liming_team/auroraX/songchentao/MyCDM/roberta/xlm-roberta-base')
        elif 'bge' in model_name.lower():
            self.model = AutoModel.from_pretrained(model_name or '/mnt/new_pfs/liming_team/auroraX/LLM/bge-large-en-v1.5')
        elif 'bert' in model_name.lower():
            self.model = BertModel.from_pretrained(model_name or '/mnt/new_pfs/liming_team/auroraX/songchentao/llama/bert-base-uncased')
        else:
            raise ValueError(f"不支持的模型类型: {model_name}，请选择 'BERT', 'RoBERTa' 或 'BGE'")
            
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        self.d_model = self.model.config.hidden_size
        # 学生能力嵌入层（u+和u-）
        self.stu_pos = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )
        # self.stu_neg = nn.Embedding(
        #     num_embeddings=num_students,
        #     embedding_dim=self.d_model
        # )
        # MLP预测头
        self.prednet = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_model, 1)
        )
        # 初始化参数
        self.initialize()

    def initialize(self):
        """参数初始化"""
        nn.init.normal_(self.stu_pos.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.stu_neg.weight, mean=0.0, std=0.1)
        # self.prednet
        for module in self.prednet:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, stu_ids, exer_in):
        # 学生的正负双模态表征
        u_pos = self.stu_pos(stu_ids)        # [batch_size, 768]
        # u_neg = self.stu_neg(stu_ids)      # [batch_size, 768]
        # 题目表征
        bert_output = self.model(            # [batch_size, 768]，提取CLS token作为题目嵌入
            input_ids=exer_in["input_ids"],
            attention_mask=exer_in["attention_mask"])
        exer_emb = bert_output.last_hidden_state[:, 0, :]

        """MLP预测头（1）不与问题交互"""
        # stu_emb = u_pos - u_neg                                       # [batch_size, 768]
        # logits = self.prednet(torch.cat([exer_emb, stu_emb], dim=1))  # [batch_size, 1]
        """MLP预测头（2）与问题交互"""
        logits = self.prednet(torch.cat([exer_emb, u_pos], dim=1))      # [batch_size, 1]
        output = torch.sigmoid(logits).squeeze(-1)                      # [batch_size]

        return output, exer_emb, u_pos, u_pos
    
    @staticmethod
    def get_loss(output, labels):
        """计算总损失"""
        preds, _, _, _ = output
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size], BCE损失
        cl_loss = torch.zeros_like(bce_loss, requires_grad=False)
        reg_loss = torch.zeros_like(bce_loss, requires_grad=False)
        total_loss = bce_loss + cl_loss + reg_loss
        return total_loss, bce_loss, cl_loss, reg_loss


class Baseline_IRT_FFT(nn.Module):
    """
    (预报头使用三参数IRT)
    使用全量微调基础模型，需要把题目文本组织为字典形式的分词结果作为模型输入
        # 全量微调模式下，所有模型参数都可以训练，不需要冻结参数
        # 默认情况下，所有参数都已经是requires_grad=True的状态
    """
    def __init__(self, num_students, model_name=None, tau=0.1, lambda_reg=1.0, lambda_cl=0.5, a_range=1.702):
        super().__init__()
        self.a_range = a_range

        # 根据选择加载不同的模型
        if 'roberta' in model_name.lower():
            self.model = XLMRobertaModel.from_pretrained(model_name or '/mnt/new_pfs/liming_team/auroraX/songchentao/MyCDM/roberta/xlm-roberta-base')
        elif 'bge' in model_name.lower():
            self.model = AutoModel.from_pretrained(model_name or '/mnt/new_pfs/liming_team/auroraX/LLM/bge-large-en-v1.5')
        elif 'bert' in model_name.lower():
            self.model = BertModel.from_pretrained(model_name or '/mnt/new_pfs/liming_team/auroraX/songchentao/llama/bert-base-uncased')
        else:
            raise ValueError(f"不支持的模型类型: {model_name}，请选择 'BERT', 'RoBERTa' 或 'BGE'")
            
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        self.d_model = self.model.config.hidden_size

        # 学生能力嵌入层
        self.stu_emb = nn.Embedding(
            num_embeddings=num_students,
            embedding_dim=self.d_model
        )
        # IRT参数映射层
        self.proj_disc = nn.Linear(self.d_model, 1)  # 区分度
        self.proj_diff = nn.Linear(self.d_model, 1)  # 难度
        self.proj_guess = nn.Linear(self.d_model, 1)  # 猜测

        # 初始化参数
        self.initialize()

    def initialize(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.stu_emb.weight)
        for module in [self.proj_disc, self.proj_diff, self.proj_guess]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.xavier_uniform_(module.bias)
        
    def forward(self, stu_ids, exer_in):
        # 学生能力表征
        theta = self.stu_emb(stu_ids)                   # [batch_size, d_model]
        # 题目表征
        bert_output = self.model(                       # [batch_size, d_model]，提取CLS token作为题目嵌入
            input_ids=exer_in["input_ids"],
            attention_mask=exer_in["attention_mask"])
        exer_emb = bert_output.last_hidden_state[:, 0]  # 等价于[:, 0, :]
        # 题目参数
        disc = torch.sigmoid(self.proj_disc(exer_emb))
        diff = self.proj_diff(exer_emb)
        guess = torch.sigmoid(self.proj_guess(exer_emb)) * 0.5

        """三参数IRT预测头"""
        output = guess + (1 - guess) / (1 + torch.exp(-self.a_range * disc * (theta - diff)))

        return output, exer_emb, theta, theta
    
    @staticmethod
    def get_loss(output, labels):
        """计算总损失"""
        preds, _, _, _ = output
        bce_loss = nn.BCELoss(reduction='mean')(preds, labels.squeeze())  # [batch_size], BCE损失
        cl_loss = torch.zeros_like(bce_loss, requires_grad=False)
        reg_loss = torch.zeros_like(bce_loss, requires_grad=False)
        total_loss = bce_loss + cl_loss + reg_loss
        return total_loss, bce_loss, cl_loss, reg_loss
