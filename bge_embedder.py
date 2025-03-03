from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json
import re
import os


# <editor-fold desc="自定义函数">


def extract_substring(text):
    """
    截取字符串字串：从第一个\"summarization\":之后起，直至第一个\"reasoning\":之前
    """
    match = re.search(r'"summarization":(.*?)"reasoning":', text, re.DOTALL)
    if match:
        return match.group(1).strip().strip(',').strip('"')
    return None


def merge_results(json_paths):
    # 读取文件，拼合结果
    output = []
    result = {}
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as file:
            temp = json.load(file)
            output.extend(temp)
    # 格式控制
    count = 0
    for elem in output:
        key = elem['idx']
        val = extract_substring(elem['output'])
        result[key] = val
        if val is None:
            count += 1
    # 检查并返回
    if count > 0:
        print('Format Error!')
    else:
        out_ind = []
        out_txt = []
        for _key, _val in result.items():
            out_ind.append(_key)
            out_txt.append(_val)
        return out_ind, out_txt


def get_embeddings(_model, _tokenizer, texts, normalize=True, batch_size=4, 
                   max_length=256, model_name="", save_path=None, resume_from=0):
    """
    获取文本嵌入向量，优化内存使用
    
    参数:
        _model: 嵌入模型
        _tokenizer: 分词器
        texts: 待嵌入的文本列表
        normalize: 是否归一化向量
        batch_size: 批处理大小
        max_length: 最大序列长度
        model_name: 模型名称
        save_path: 中间结果保存路径，用于断点续传
        resume_from: 从第几个样本继续处理
    """
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 仅将最必要的组件移动到GPU
    _model = _model.to(device)
    _model.eval()  # 评估模式
    
    # 如果有中间结果，加载它们
    if save_path and os.path.exists(save_path) and resume_from > 0:
        try:
            saved_embeddings = np.load(save_path)
            embeddings_list = [saved_embeddings]
            start_idx = resume_from
            print(f"继续从第 {start_idx} 个样本处理...")
        except Exception as e:
            print(f"无法加载中间结果: {e}")
            embeddings_list = []
            start_idx = 0
    else:
        embeddings_list = []
        start_idx = 0
    
    # 计算总批次数以便显示进度
    total_batches = (len(texts) - start_idx + batch_size - 1) // batch_size
    
    try:
        # 分批处理文本
        for i in range(start_idx, len(texts), batch_size):
            # 显示进度
            current_batch = (i - start_idx) // batch_size + 1
            if current_batch % 100 == 0:
                print(f'处理批次 {current_batch}/{total_batches} ({(current_batch/total_batches*100):.1f}%)')
            
            # 获取当前批次文本
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            
            # 模型特定的预处理
            if "bge" in model_name.lower():
                if "zh" in model_name.lower():
                    batch_texts = [f"为这个句子生成表示：{text}" for text in batch_texts]
                else:
                    batch_texts = [f"Represent this sentence: {text}" for text in batch_texts]
            
            # 编码文本，使用较小的max_length减少内存使用
            encoded_input = _tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # 将输入移到GPU，只在需要时移动
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # 减少内存使用的关键策略
            with torch.no_grad():
                # 使用torch.cuda.amp.autocast()可以降低精度，减少内存
                if device == 'cuda':
                    with torch.cuda.amp.autocast(enabled=True):
                        model_output = _model(**encoded_input)
                else:
                    model_output = _model(**encoded_input)
                
                # 获取嵌入并立即转移到CPU
                batch_embeddings = model_output.last_hidden_state[:, 0].cpu()
                
                # 如果需要归一化，在CPU上进行
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # 转换为numpy并立即释放张量
                batch_numpy = batch_embeddings.numpy()
                del batch_embeddings
                embeddings_list.append(batch_numpy)
            
            # 立即释放GPU内存
            del encoded_input
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # 每处理一定数量的批次保存中间结果
            if save_path and current_batch % 100 == 0:
                temp_embeddings = np.vstack(embeddings_list)
                np.save(f"{save_path}_temp_{i}", temp_embeddings)
                print(f"保存临时结果到 {save_path}_temp_{i}")
    
    except Exception as e:
        print(f"处理中发生错误: {e}")
        # 发生错误时保存已处理的结果
        if save_path and embeddings_list:
            temp_embeddings = np.vstack(embeddings_list)
            np.save(f"{save_path}_error_at_{i}", temp_embeddings)
            print(f"错误发生前的结果已保存到 {save_path}_error_at_{i}")
        raise e
    
    finally:
        # 确保模型回到CPU，释放GPU内存
        _model = _model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 合并所有批次的嵌入
    if embeddings_list:
        _embeddings = np.vstack(embeddings_list)
        
        # 保存最终结果
        if save_path:
            np.save(save_path, _embeddings)
            print(f"完整结果已保存到 {save_path}")
            
        return _embeddings
    else:
        return None


# 文本嵌入函数
def get_embeddings_backup(_model, _tokenizer, texts, normalize=True, batch_size=8):
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)

    # 将模型设置为评估模式
    _model.eval()

    embeddings_list = []

    # 分批处理，避免内存溢出
    for i in range(0, len(texts), batch_size):
        if (i+1) % 100 == 0:
            print(f'{i+1} of {len(range(0, len(texts), batch_size))}')
        
        if i == max(range(0, len(texts), batch_size)):
            batch_texts = texts[i: i + batch_size]
        else:
            batch_texts = texts[i: len(texts)]

        # BGE模型推荐的特殊预处理
        if "bge" in model_name:
            if "zh" in model_name:
                # 中文模型添加特殊前缀
                batch_texts = [f"为这个句子生成表示：{text}" for text in batch_texts]
            else:
                # 英文模型添加特殊前缀
                batch_texts = [f"Represent this sentence: {text}" for text in batch_texts]

        # 编码文本
        encoded_input = _tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        # 获取嵌入向量
        with torch.no_grad():
            model_output = _model(**encoded_input)
            # BGE模型使用[CLS]令牌的输出作为句子嵌入
            batch_embeddings = model_output.last_hidden_state[:, 0]

            if normalize:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            embeddings_list.append(batch_embeddings.cpu().numpy())

    # 合并所有批次的嵌入
    _embeddings = np.vstack(embeddings_list)
    return _embeddings


def calculate_similarity(_embeddings):
    """
    计算学生描述之间的相似度
    """
    _similarity_matrix = np.zeros((len(_embeddings), len(_embeddings)))
    for i in range(len(_embeddings)):
        for j in range(len(_embeddings)):
            # 计算余弦相似度
            similarity = np.dot(_embeddings[i], _embeddings[j])
            _similarity_matrix[i][j] = similarity
    return _similarity_matrix


# </editor-fold>


if __name__ == '__main__':

    # <editor-fold desc="模型和数据集基本参数">
    # Number of Students, Number of Exercises, Number of Knowledge Concepts
    n_stu = 6148  # 1 - 6147，共4918人（非致密）
    n_exer = 948  # 0-947，共948题（致密）
    n_conc = 656  # 3-1988，共388个KC（非致密），其中3-655是题库涉及的KC（实际操作时将根部的两层KC剔除）

    # 选择合适的BGE模型（以下选项任选其一）
    # model_name = "BAAI/bge-small-zh-v1.5"  # 小型中文模型
    # model_name = "BAAI/bge-base-zh-v1.5"   # 基础中文模型
    # model_name = "BAAI/bge-large-zh-v1.5"  # 大型中文模型
    model_name = "/mnt/new_pfs/liming_team/auroraX/LLM/bge-large-en-v1.5"  # 英文模型
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # </editor-fold>

    # <editor-fold desc="学生知识水平描述嵌入">

    root_dir = '/mnt/new_pfs/liming_team/auroraX/songchentao/gpt/KCD'
    task_name = 'sct_all_diag_user_20Clip_'
    start_idx = range(0, 5000, 500)
    stu_jsons = [f'{root_dir}/response_{task_name}_{e}_{e+500}.json' for e in start_idx]
    stu_ids, student_descriptions = merge_results(stu_jsons)  # dict{idx: description}
    # 最终的学生描述示例：
    # student_descriptions = [
    #     "该学生在数学方面表现出色，尤其是代数和微积分概念掌握扎实。然而，在几何证明方面仍需加强。物理基础知识良好，但对于电磁学的理解有待提高。学生展现出较强的逻辑思维能力，但有时缺乏创新性思考。",
    #     "这位学生的语文素养很高，阅读理解能力强，写作生动有创意。英语听说读写均衡发展，能流利表达。历史和政治科目有深入思考，但地理知识薄弱。该生自主学习能力强，善于整合跨学科知识。",
    #     "该学生编程能力突出，已掌握Python和Java基础，能独立完成小型项目。数据结构理解到链表和树的层次，但算法思维有待提升。数学基础扎实，统计学概念清晰。该生学习主动性高，但时间管理能力需要改进。"
    # ]

    # # 获取嵌入向量
    # embeddings = get_embeddings(model, tokenizer, student_descriptions, normalize=True, batch_size=1,
    #                             max_length=256, model_name=model_name, save_path=None, resume_from=0)
    # # 打印嵌入向量的形状
    # print(f"嵌入向量形状: {embeddings.shape}")  # (n_stu_dense, 1024)
    # embeddings = embeddings.astype(np.float32)  # 对齐数据格式

    # # 转存嵌入矩阵
    # stu_emb = np.zeros((n_stu, 1024), dtype=np.float32)
    # for idx, stu_id in enumerate(stu_ids):
    #     stu_emb[stu_id, :] = embeddings[idx, :]
    # np.save(f'{root_dir}/user_emb_{task_name}.npy', stu_emb)
    stu_emb = np.load(f'{root_dir}/user_emb_{task_name}.npy')

    """计算并显示相似度矩阵"""
    similarity_matrix = calculate_similarity(stu_emb)
    print("\n学生知识描述相似度矩阵:")
    print(similarity_matrix)
    dict_sim = {}
    # 转存相似度信息为
    for ind_row in stu_ids:
        temp = similarity_matrix[ind_row, stu_ids].tolist()
        # 排序并取负号实现倒序，获取前20个最大值的索引
        dict_sim[str(ind_row)] = np.argsort(temp)[-20:][::-1].tolist()
    with open(f"{root_dir}/similar_students.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_sim, json_file, indent=4, ensure_ascii=False)

    # </editor-fold>

    # <editor-fold desc="题目描述嵌入">

    root_dir = '/mnt/new_pfs/liming_team/auroraX/songchentao/gpt/KCD'
    task_name = 'sct_all_diag_exer_20Clip_'
    start_idx = range(0, 1000, 500)
    exer_jsons = [f'{root_dir}/response_{task_name}_{e}_{e + 500}.json' for e in start_idx]
    exer_ids, exer_descriptions = merge_results(exer_jsons)  # dict{idx: description}

    # 获取嵌入向量
    embeddings = get_embeddings(model, tokenizer, exer_descriptions, normalize=True, batch_size=1,
                                max_length=256, model_name=model_name, save_path=None, resume_from=0)
    # 打印嵌入向量的形状
    print(f"嵌入向量形状: {embeddings.shape}")    # (n_stu_dense, 1024)
    embeddings = embeddings.astype(np.float32)  # 对齐数据格式

    # 转存嵌入矩阵
    exer_emb = np.zeros((n_exer, 1024), dtype=np.float32)
    for idx, exer_id in enumerate(exer_ids):
        exer_emb[exer_id, :] = embeddings[idx, :]
    np.save(f'{root_dir}/item_emb_{task_name}.npy', exer_emb)

    """计算并显示相似度矩阵"""
    similarity_matrix = calculate_similarity(exer_emb)
    print("\n题目描述相似度矩阵:")
    print(similarity_matrix)
    dict_sim = {}
    # 转存相似度信息为
    for ind_row in exer_ids:
        temp = similarity_matrix[ind_row, exer_ids].tolist()
        # 排序并取负号实现倒序，获取前20个最大值的索引
        dict_sim[str(ind_row)] = np.argsort(temp)[-20:][::-1].tolist()
    with open(f"{root_dir}/similar_exercises.json", 'w', encoding='utf-8') as json_file:
        json.dump(dict_sim, json_file, indent=4, ensure_ascii=False)

    # </editor-fold>

    # <editor-fold desc="其他功能">
    """示例应用：找到与目标学生描述最相似的描述"""
    # target_description = "这位学生数学基础牢固，尤其擅长解决应用题，但几何证明较弱。具有良好的逻辑分析能力，物理中力学部分掌握较好，但电磁学概念模糊。学习态度积极，但需要提高系统化思考能力。"
    # # 获取目标描述的嵌入
    # target_embedding = get_embeddings(model, tokenizer, [target_description], normalize=True)[0]
    # # 计算与其他学生描述的相似度
    # similarities = [np.dot(target_embedding, emb) for emb in embeddings]
    # most_similar_idx = np.argmax(similarities)
    # # 打印结果
    # print(f"\n与目标学生描述最相似的是学生描述 #{most_similar_idx + 1}")
    # print(f"相似度分数: {similarities[most_similar_idx]:.4f}")
    # </editor-fold>

