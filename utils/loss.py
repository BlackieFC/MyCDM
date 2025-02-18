import torch


def custom_loss(q, u_plus, u_minus, y, tau, norm=False):
    """
    自定义对比损失函数：
    Loss = -\frac{1}{N}\sum_{i=1}^{N}{[y_{i}log\frac{exp(\frac{s(q_{i}, u^{+})}{\tau})}{exp(\frac{s(q_{i}, u^{+})}{\tau}) +
    exp(\frac{s(q_{i}, u^{-})}{\tau})} + (1-y_{i}) log\frac{exp(\frac{s(q_{i}, u^{-})}{\tau})}{exp(\frac{s(q_{i}, u^{+})}{\tau}) +
    exp(\frac{s(q_{i}, u^{-})}{\tau})}]} +
    \lambda\cdot\left| \frac{u^{+}\cdot u^{-}}{\left| \left| u^{+} \right| \right| \left| \left| u^{-} \right| \right|} \right|
    """
    # 归一化数据，保证各项loss的取值范围在0-1以内
    if norm:
        q = q / q.norm(dim=-1, keepdim=True)
        u_plus = u_plus / u_plus.norm(dim=-1, keepdim=True)
        u_minus = u_minus / u_minus.norm(dim=-1, keepdim=True)

    # 计算正负样本相似度（点积）并缩放
    s_pos = torch.sum(q * u_plus, dim=1) / tau                         # (bs,)
    s_neg = torch.sum(q * u_minus, dim=1) / tau                        # (bs,)

    # 计算对数概率（数值稳定版）
    log_sum_exp = torch.logsumexp(torch.stack([s_pos, s_neg]), dim=0)  # (2,bs) -> (bs,)
    log_prob_pos = s_pos - log_sum_exp                                 # log(exp(s_pos)/sum)
    log_prob_neg = s_neg - log_sum_exp                                 # log(exp(s_neg)/sum)

    # 对比损失项
    term_per_sample = y * log_prob_pos + (1 - y) * log_prob_neg
    loss_contrast = -term_per_sample.mean()

    # 正则化项（余弦相似度绝对值均值）
    dot_product = torch.sum(u_plus * u_minus, dim=1)                   # (bs,)
    norm_plus = torch.norm(u_plus, p=2, dim=1)                         # (bs,)
    norm_minus = torch.norm(u_minus, p=2, dim=1)                       # (bs,)
    cos_sim = dot_product / (norm_plus * norm_minus + 1e-8)            # 防止除零
    loss_reg = torch.abs(cos_sim).mean()

    # 返回对比损失&正则化损失
    return loss_contrast, loss_reg
