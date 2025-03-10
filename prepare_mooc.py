# NIPS34数据的预处理脚本
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os
import ast
from collections import Counter
import warnings
from itertools import chain
import networkx as nx
import math
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略FutureWarning类型的警告




# <editor-fold desc="自定义函数">


def split_list_by_ratio(input_list, ratios, seed=None):
    if seed is not None:
        random.seed(seed)
    assert sum(ratios) == 1.0, "Ratios must sum to 1"
    random.shuffle(input_list)       # Shuffle the input list to ensure randomness
    total_length = len(input_list)
    splits = []
    start_index = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:  # Ensure the last split gets the remaining elements
            split_length = total_length - start_index
        else:
            split_length = int(total_length * ratio)

        splits.append(input_list[start_index:start_index + split_length])
        start_index += split_length
    return splits


def split_lists_by_index(lists, fraction=0.2, seed=None):
    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    # 取 first list 的长度
    list_length = len(lists[0])

    # 随机抽取 20% 的索引
    num_indices_to_select = round(list_length * fraction)
    selected_indices = set(random.sample(range(list_length), num_indices_to_select))

    # 初始化两个部分的列表
    lists_part1 = [[] for _ in lists]
    lists_part2 = [[] for _ in lists]

    # 根据索引分割每个列表
    for i in range(list_length):
        if i in selected_indices:
            for j, lst in enumerate(lists):
                lists_part1[j].append(lst[i])
        else:
            for j, lst in enumerate(lists):
                lists_part2[j].append(lst[i])

    return lists_part1, lists_part2


def add_similarity_edges(tree):
    """
    在一个有向树形图的基础上，添加一些无向边。
    其中这些无向边连接同一父节点下最大深度2以内的节点，
    但这些节点原本之间没有“先验”关系
    """
    # 创建树形图的副本，我们不想直接修改原图
    graph = tree.copy()

    # 找到每个节点的深度
    depths = nx.shortest_path_length(tree, source=None)

    # 将生成器对象转换为字典
    depths = {node: depth_dict for node, depth_dict in depths}

    for parent in tree.nodes():
        # 获取以parent为根的子树的所有节点，且深度在2以内
        nodes_at_depth_1_or_2 = []
        for node in nx.descendants(tree, parent):
            if node in depths[parent] and depths[parent][node] in {1, 2}:  # 修改此行以匹配字典结构
                nodes_at_depth_1_or_2.append(node)

        # 为这些节点添加“相似”无向边
        for i in range(len(nodes_at_depth_1_or_2)):
            for j in range(i + 1, len(nodes_at_depth_1_or_2)):
                node1, node2 = nodes_at_depth_1_or_2[i], nodes_at_depth_1_or_2[j]
                # 确保他们之间没有“先验”关系后，添加“相似”无向边
                if not tree.has_edge(node1, node2) and not tree.has_edge(node2, node1):
                    graph.add_edge(node1, node2, relation='similar')

    return graph


def add_similarity_edges_depth1(tree):
    """
    在一个有向树形图的基础上，添加一些无向边。
    其中这些无向边连接同一父节点下的子节点，这些子节点之间没有直接的父子关系。
    """
    # 创建树形图的副本，我们不想直接修改原图
    graph = tree.copy()

    # 遍历每个节点，找到其子节点
    for parent in tree.nodes():
        children = list(tree.successors(parent))  # 获取直接子节点

        # 为这些子节点添加“相似”无向边
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                node1, node2 = children[i], children[j]
                # 确保他们之间没有直接的父子关系后，添加“相似”无向边
                if not tree.has_edge(node1, node2) and not tree.has_edge(node2, node1):
                    graph.add_edge(node1, node2, relation='similar')

    return graph


def add_similarity_edges_depth1_betweenLeaves(tree):
    """
    在一个有向树形图的基础上，添加一些无向边。
    其中这些无向边连接同一父节点下的子节点，这些子节点之间没有直接的父子关系。
    仅当两个子节点均为叶子节点时，添加无向边。
    """
    # 创建树形图的副本，我们不想直接修改原图
    graph = tree.copy()

    # 遍历每个节点，找到其子节点
    for parent in tree.nodes():
        children = list(tree.successors(parent))  # 获取直接子节点

        # 为这些子节点添加“相似”无向边
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                node1, node2 = children[i], children[j]
                # 确保他们之间没有直接的父子关系后，添加“相似”无向边
                if (not tree.has_edge(node1, node2) and
                        not tree.has_edge(node2, node1) and
                        tree.out_degree(node1) == 0 and
                        tree.out_degree(node2) == 0):
                    graph.add_edge(node1, node2, relation='similar')

    return graph


def flatten(lst):
    while any(isinstance(i, list) for i in lst):
        lst = list(chain.from_iterable(lst))
    return lst


def remove_substring_and_after(original_str, target):
    index = original_str.find(target)
    if index != -1:
        return original_str[:index]
    return original_str


def is_first_attempt(lst):
    index_dict = {}  # 用于记录每个元素第一次出现的索引
    result = []  # 保存结果的列表
    for i, element in enumerate(lst):
        if element not in index_dict.keys():
            index_dict[element] = i  # 记录第一次出现的位置
        result.append(index_dict[element])  # 添加到结果列表中
    return result


def slice_by_indices(original_list, indices):
    sliced_list = [original_list[index] for index in indices]
    return sliced_list


def reset_seq():
    sour = {key: [] for key in cols_raw}
    targ = {key: [] for key in cols_raw}
    return sour, targ


# </editor-fold>




# <editor-fold desc="NIPS34">

"""参数设置"""
random_seed = 42  # 为None时不乱序，取最高频的10%进行分割
thres_exer = 100  # 题目在整个数据集中最少出现的交互次数
ratio_exer = 0.3  # target占比（新题目）
ratio_user = 0.3  # target占比（新学生）
thres_conc = 100  # 同上，针对新KC情景
ratio_conc = 0.3
path_pics = './pics'
name_data = 'NIPS34'

"""读取数据，统计数量"""
df = pd.read_csv(r'C:\Games\NIPS\data\train_data\train_task_3_4.csv',
                 usecols=['QuestionId', 'UserId', 'IsCorrect'])
df_Qmat = pd.read_csv(r'C:\Games\NIPS\data\metadata\question_metadata_task_3_4.csv')
df = df.merge(df_Qmat[['QuestionId', 'SubjectId']], on='QuestionId', how='left')  # 共138w条交互
df_KC_tree = pd.read_csv(r'C:\Games\NIPS\data\metadata\subject_metadata.csv')

# 获取学生ID
n_stu = df['UserId'].value_counts()  # 4918
sids = [sid for sid, _ in n_stu.items()]  # 1-6147，共4918人
sids.sort()
n_stu = len(sids)

# 获取问题ID
n_exer = df['QuestionId'].value_counts()  # 948
pids = [pid for pid, _ in n_exer.items()]  # 0-947，共948题（致密）
pids.sort()
n_exer = len(pids)

# 获取知识点ID
kids = df_KC_tree['SubjectId'].tolist()
kids.sort()
n_conc = len(kids)  # 388个KC，ID从3到1988

# check
import ast
result = set()
for ind, row in df_Qmat.iterrows():
    temp = ast.literal_eval(row['SubjectId'])
    result.update(temp)




# 将df整合为指定格式
col_names = ['uid', 'questions', 'concepts', 'responses']
df = df.sort_values(['UserId'], ascending=True).reset_index(drop=True)
df_user = pd.DataFrame(columns=col_names)
for sid in sids:
    dict_s = {elem: [] for elem in col_names}
    dict_s['uid'].append(str(sid))
    df_slice = df[df['UserId'] == sid]
    for _ind, row in df_slice.iterrows():
        dict_s['questions'].append(str(row["QuestionId"]))
        temp = ast.literal_eval(row["SubjectId"])
        temp = list(map(str, temp))
        dict_s['concepts'].append('_'.join(temp))
        dict_s['responses'].append(str(row["IsCorrect"]))
    for key, value in dict_s.items():
        dict_s[key] = ','.join(value)
    dict_s = pd.DataFrame(dict_s, index=[0])
    df_user = pd.concat([df_user, dict_s], ignore_index=True)

df = df_user  # 4918,4

# </editor-fold>




# <editor-fold desc="新题目">

"""柱状图：题目视角 —— Y: problem_id, X:count_student"""
counter = []
for ind, row in df.iterrows():
    # question无需去重（question-level）
    _temp_list = list(map(int, row['questions'].split(',')))
    _temp_isre = list(map(int, row['responses'].split(',')))
    zero_indices = [index for index, value in enumerate(_temp_isre) if value != -1]
    # question需要去重（kc-level）
    # _temp_isre = list(map(int, row['is_repeat'].split(',')))
    # zero_indices = [index for index, value in enumerate(_temp_isre) if value == 0]
    counter.extend([_temp_list[index] for index in zero_indices])

counter = Counter(counter)
df_pid = pd.DataFrame(list(counter.items()), columns=['problem_id', 'Count'])

ds = df_pid.sort_values(['Count'], ascending=False).reset_index(drop=True)
# plot_dataframe_bar_chart_rot90(ds,
#                                'problem_id',
#                                'Count',
#                                'All problems by counts(descending)',
#                                name_data,
#                                path_pics,
#                                n_xtick=10,
#                                line=True
#                                )

# 切分掉交互次数过少的题目（因为还要训练Oracle模型）
ds = ds[ds['Count'] >= thres_exer]
n_exer = len(ds)
# 使用固定随机种子乱序，并进行切分；不乱序时，则切分的是频率最高的题目（约10%）和低频题目
if random_seed is not None:
    ds = ds.sample(frac=1, random_state=random_seed).reset_index(drop=True)
pid_source = ds["problem_id"].tolist()
pid_source = pid_source[:int(-n_exer * ratio_exer)]
pid_target = ds["problem_id"].tolist()
pid_target = pid_target[int(-n_exer * ratio_exer):]
n_exer_s = len(pid_source)
n_exer_t = len(pid_target)

# 初始化输出用的dataframe
cols = ['user_id', 'exer_id', 'knowledge_code', 'score']  # 'domain'
cols_raw = col_names
df_source = pd.DataFrame(columns=cols_raw)
df_target = pd.DataFrame(columns=cols_raw)

# 排序（已排好）
# df = df.sort_values(['uid','timestamps']).reset_index(drop=True)

"""question-level：直接写入"""
seq_source, seq_target = reset_seq()
last_uid = -1
for ind, row in df.iterrows():
    if (ind + 1) % 100 == 0:
        print('{} of {}'.format(ind + 1, len(df)))

    curr_uid = int(row['uid'])
    if curr_uid != last_uid and last_uid != -1:
        # 处理上一学生（去重）
        fir_att_s = is_first_attempt(seq_source['questions'])
        fir_att_t = is_first_attempt(seq_target['questions'])
        seq_source = {key: slice_by_indices(value, fir_att_s) for key, value in seq_source.items()}
        seq_target = {key: slice_by_indices(value, fir_att_t) for key, value in seq_target.items()}
        seq_source['uid'] = seq_source['uid'][:1]
        seq_target['uid'] = seq_target['uid'][:1]
        seq_source = {key: ','.join(list(map(str, value))) for key, value in seq_source.items()}
        seq_target = {key: ','.join(list(map(str, value))) for key, value in seq_target.items()}
        seq_source = pd.DataFrame(seq_source, index=[0])
        seq_target = pd.DataFrame(seq_target, index=[0])
        # 保存上一学生（将状态记录字典写入）
        df_source = pd.concat([df_source, seq_source], ignore_index=True)
        df_target = pd.concat([df_target, seq_target], ignore_index=True)
        # 重置新学生状态记录字典
        seq_source, seq_target = reset_seq()

    # （无论如何执行）拆分当前行的数据为source、target后，向记录状态字典中写入当前行的数据
    _c = row['concepts'].split(',')
    _r = row['responses'].split(',')
    for _ind, _q in enumerate(list(map(int, row['questions'].split(',')))):
        if _q in pid_source:
            seq_source['uid'].append(curr_uid)
            seq_source['questions'].append(_q)
            seq_source['concepts'].append(_c[_ind])
            seq_source['responses'].append(_r[_ind])
        elif _q in pid_target:
            seq_target['uid'].append(curr_uid)
            seq_target['questions'].append(_q)
            seq_target['concepts'].append(_c[_ind])
            seq_target['responses'].append(_r[_ind])
        elif _q == -1:
            break
    # （无论如何执行）更新last_uid，准备下一次迭代
    last_uid = curr_uid

# 处理最后一学生（去重）
fir_att_s = is_first_attempt(seq_source['questions'])
fir_att_t = is_first_attempt(seq_target['questions'])
seq_source = {key: slice_by_indices(value, fir_att_s) for key, value in seq_source.items()}
seq_target = {key: slice_by_indices(value, fir_att_t) for key, value in seq_target.items()}
seq_source['uid'] = seq_source['uid'][:1]
seq_target['uid'] = seq_target['uid'][:1]
seq_source = {key: ','.join(list(map(str, value))) for key, value in seq_source.items()}
seq_target = {key: ','.join(list(map(str, value))) for key, value in seq_target.items()}
seq_source = pd.DataFrame(seq_source, index=[0])
seq_target = pd.DataFrame(seq_target, index=[0])
# 保存最后一学生（将状态记录字典写入）
df_source = pd.concat([df_source, seq_source], ignore_index=True)
df_target = pd.concat([df_target, seq_target], ignore_index=True)

# <editor-fold desc="新题目：QC（针对source）">

# 该情景下，需要target域中没有“新学生”（不增加额外的难度）
set_uid = set(df_source['uid'].unique())  # 18066

# 记录结果用（如果需要限制target域上的整体表现的话）
cols = ['u_id', 'num_target', 'num_correct', 'cor_target']
df_result = pd.DataFrame(columns=cols)

for ind, uid in enumerate(set_uid):
    if (ind + 1) % 1000 == 0:
        print("{} of {}".format(ind + 1, len(set_uid)))

    sliced_source = df_source[df_source['uid'] == uid].reset_index()  # 一个一个学生地看
    """适用于question-based"""
    temp = sliced_source.loc[0, 'responses']
    temp = temp.split(',')
    if len(temp) < 20:
        df_source.drop(df_source[df_source['uid'] == uid].index, inplace=True)
        df_target.drop(df_target[df_target['uid'] == uid].index, inplace=True)
    """适用于record-based"""
    # sliced_target = df_target[df_target['u_id'] == uid]
    # # 剔除不满足条件的行
    # if len(sliced_source) < 20:
    #     df_source.drop(df_source[df_source['u_id'] == uid].index, inplace=True)
    #     df_target.drop(df_target[df_target['u_id'] == uid].index, inplace=True)
    # 计算
    # _score_tar = list(map(int, sliced_target['score'].tolist()))
    # _num_tar = len(sliced_target)    # target 记录总数
    # _num_cor = sum(_score_tar)       # target 答对数
    # if _num_tar > 0:
    #     _cor_tar = _num_cor / _num_tar   # target 正确率
    # else:
    #     _cor_tar = None
    # # 记录
    # _data = [uid, _num_tar, _num_cor, _cor_tar]
    # df_result = pd.concat([df_result, pd.DataFrame([pd.Series(_data, index=cols)])], ignore_index=True)

# 最终保存
df_source.to_csv('source_exer_nips34.csv', index=False)
df_target.to_csv('target_exer_nips34.csv', index=False)

"""用于控制数据在测试集上的整体表现"""
# # 统计
# df_result = df_result.sort_values(['cor_target', 'num_correct']).reset_index(drop=True)
# df_result_ = df_result  # backup
# # 切片
# ratio_double_tail = 0.1
# df_result = df_result.head(int((1-ratio_double_tail)*len(df_result))).tail(int((1-ratio_double_tail)*len(df_result)))
# _corr = 0
# for ind in range(1, len(df_result)):
#     _temp = df_result.head(ind)
#     _corr = sum(_temp['num_correct'].tolist()) / sum(_temp['num_target'].tolist())
#     if _corr > 0.8:
#         print(ind)
#         break
# print(_corr)  # 0.7548
# uid_set = list(map(int, df_result['u_id'].tolist()))
# # 剔除A中C列内容未出现于列表B中的所有行
# df_source_f = df_source[df_source['u_id'].isin(uid_set)]
# df_target_f = df_target[df_target['u_id'].isin(uid_set)]
# # 最终保存
# df_source_f.to_csv('source_exer_xes.csv', index=False)
# df_target_f.to_csv('target_exer_xes.csv', index=False)

# </editor-fold>

# </editor-fold>




# <editor-fold desc="新学生">

# 声明处理后的Dataframe
# cols_raw = ['uid', 'questions', 'concepts', 'responses', 'len']
# df_total = pd.DataFrame(columns=cols_raw)

set_uid = set(df['uid'].unique())  # 18066
n_user = len(set_uid)

# for ind, uid in enumerate(set_uid):
#     if (ind+1) % 1000 == 0:
#         print("{} of {}".format(ind+1, n_user))
#     # 一个一个学生地看
#     sliced_source = df[df['uid'] == uid].reset_index(drop=True)
#     # 按照时间顺序排序
#     sliced_source = sliced_source.sort_values(['timestamps']).reset_index(drop=True)
#     # 拼合 & 剔除padding值，并返回有效长度
#     _questions = []
#     _concepts = []
#     _responses = []
#     for _ind, row in sliced_source.iterrows():
#         _questions.append(remove_substring_and_after(row['questions'], ',-1'))
#         _concepts.append(remove_substring_and_after(row['concepts'], ',-1'))
#         _responses.append(remove_substring_and_after(row['responses'], ',-1'))
#     length = ','.join(_responses)
#     length = len(length.split(','))
#     new_row = {'uid': uid,
#                'questions': ','.join(_questions),
#                'concepts': ','.join(_concepts),
#                'responses': ','.join(_responses),
#                'len': length
#                }
#     df_total = df_total.append(new_row, ignore_index=True)  # 反正一行一行写入，就append吧
#     """XES数据都很长，无需QC"""
#     pass

df_total = df

"""指定的random seed下拆分为source和target"""
# 使用固定随机种子乱序，并进行切分；
if random_seed is not None:
    df_total = df_total.sample(frac=1, random_state=random_seed).reset_index(drop=True)
# 拆分学生ID和Dataframe
df_source = df_total.iloc[:int(-n_user * ratio_user)]  # Dataframe
df_target = df_total.iloc[int(-n_user * ratio_user):]
uid_source = df_total["uid"].tolist()  # ID列表
uid_source = uid_source[:int(-n_user * ratio_user)]
uid_target = df_total["uid"].tolist()
uid_target = uid_target[int(-n_user * ratio_user):]
n_user_s = len(uid_source)
n_user_t = len(uid_target)

# 保存结果
df_source.to_csv('source_user_nips34.csv', index=False)
df_target.to_csv('target_user_nips34.csv', index=False)

# </editor-fold>




# <editor-fold desc="新KC">

"""不可随意进行乱序，部分知识点涉及过于频繁"""
domain_32 = []  # Number
domain_71 = []  # Geometry
for ind, row in df_Qmat.iterrows():
    temp = ast.literal_eval(row['SubjectId'])
    if temp[1] == 32:
        domain_32.extend(temp[2:])
    elif temp[1] == 71:
        domain_71.extend(temp[2:])
domain_32 = set(domain_32)  # 28
domain_71 = set(domain_71)  # 35

cid_source = domain_32
cid_target = domain_71
n_conc_s = len(cid_source)  # 28
n_conc_t = len(cid_target)  # 35
n_conc = n_conc_s + n_conc_t  # 63

# 重新读取文件
df_total = pd.read_csv(r'C:\Games\NIPS\data\train_data\train_task_3_4.csv',
                       usecols=['QuestionId', 'UserId', 'IsCorrect'])
df_Qmat = pd.read_csv(r'C:\Games\NIPS\data\metadata\question_metadata_task_3_4.csv')
df_total = df_total.merge(df_Qmat[['QuestionId', 'SubjectId']], on='QuestionId', how='left')  # 共138w条交互
df_total['temp'] = df_total['SubjectId']
df_total['SubjectId_'] = df_total['SubjectId']
df_total['SubjectId'] = df_total['SubjectId'].apply(lambda x: '[' + x[8:])
df_total['temp'] = df_total['temp'].apply(lambda x: x[4:6])

# 重命名
df_total.rename(columns={
    'UserId': 'uid',
    'QuestionId': 'questions',
    'SubjectId': 'concepts',
    'IsCorrect': 'responses'
}, inplace=True)

# # total: 将df整合为指定格式
# col_names = ['uid', 'questions', 'concepts', 'responses']
# df_temp = df_total.sort_values(['uid'],ascending=True).reset_index(drop=True)
# df_user = pd.DataFrame(columns=col_names)
# for sid in sids:
#     dict_s = {elem: [] for elem in col_names}
#     dict_s['uid'].append(str(sid))
#     df_slice = df_temp[df_temp['uid'] == sid]
#     for _ind, row in df_slice.iterrows():
#         dict_s['questions'].append(str(row["questions"]))
#         temp = ast.literal_eval(row["concepts"])
#         temp = list(map(str, temp))
#         dict_s['concepts'].append('_'.join(temp))
#         dict_s['responses'].append(str(row["responses"]))
#     """检查学生的交互数是否满足"""
#     if len(dict_s['questions']) < 5:
#         continue
#     for key, value in dict_s.items():
#         dict_s[key] = ','.join(value)
#     dict_s = pd.DataFrame(dict_s, index=[0])
#     df_user = pd.concat([df_user, dict_s], ignore_index=True)
#
# df_user.to_csv('total.csv', index=False)


df_source = df_total[df_total['temp'] == '32']
df_target = df_total[df_total['temp'] == '71']

"""
250113新增：用于确保target中的学生必须在source有数据，同时source中的train和val的学生必须保持一致
"""
uid_s = df_source['uid'].value_counts()  # 4806
uid_s = uid_s[uid_s >= 15]               # 4702
uid_s = uid_s.index.tolist()
uid_s.sort()
uid_t = df_target['uid'].value_counts()  # 4600
uid_t = uid_t[uid_t >= 15]               # 3945
uid_t = uid_t.index.tolist()
uid_t.sort()
# 剔除uid_t中不存在于uid_s的元素
uid_t = [elem for elem in uid_t if elem in uid_s]  # 3744
# 这样，剩下的3744名学生在source和target域上均有至少15道做题记录

"""
source: 将df整合为指定格式
    ——250113:注意保证train 和 val中的学生完全一致
"""
col_names = ['uid', 'questions', 'concepts', 'responses']
df_source = df_source.sort_values(['uid'], ascending=True).reset_index(drop=True)
df_user = pd.DataFrame(columns=col_names)  # 声明输出
# for sid in sids:
for sid in uid_s:   # 注意替换了学生范围（保证学生在source上至少有15条记录）
    dict_s = {elem: [] for elem in col_names}
    dict_s['uid'].append(str(sid))                         # 记录当前uid
    df_slice = df_source[df_source['uid'] == sid]          # 切片该学生在source域上交互
    for _ind, row in df_slice.iterrows():
        dict_s['questions'].append(str(row["questions"]))  #
        temp = ast.literal_eval(row["concepts"])
        temp = list(map(str, temp))
        dict_s['concepts'].append('_'.join(temp))
        dict_s['responses'].append(str(row["responses"]))
    """检查学生的交互数是否满足"""
    if len(dict_s['questions']) < 5:
        continue
    for key, value in dict_s.items():
        dict_s[key] = ','.join(value)
    dict_s = pd.DataFrame(dict_s, index=[0])
    df_user = pd.concat([df_user, dict_s], ignore_index=True)

df_user.to_csv('source_conc_nips34.csv', index=False)

# target: 将df整合为指定格式
col_names = ['uid', 'questions', 'concepts', 'responses']
df_target = df_target.sort_values(['uid'], ascending=True).reset_index(drop=True)
df_user = pd.DataFrame(columns=col_names)
# for sid in sids:
for sid in uid_t:   # 注意替换了学生范围（保证学生在target上至少有15条记录）
    dict_s = {elem: [] for elem in col_names}
    dict_s['uid'].append(str(sid))
    df_slice = df_target[df_target['uid'] == sid]
    for _ind, row in df_slice.iterrows():
        dict_s['questions'].append(str(row["questions"]))
        temp = ast.literal_eval(row["concepts"])
        temp = list(map(str, temp))
        dict_s['concepts'].append('_'.join(temp))
        dict_s['responses'].append(str(row["responses"]))
    """检查学生的交互数是否满足"""
    if len(dict_s['questions']) < 5:
        continue
    for key, value in dict_s.items():
        dict_s[key] = ','.join(value)
    dict_s = pd.DataFrame(dict_s, index=[0])
    df_user = pd.concat([df_user, dict_s], ignore_index=True)

df_user.to_csv('target_conc_nips34.csv', index=False)

# </editor-fold>




"""
prepare_data部分之前由于误操作清除了，但不影响，需要时补充即可
"""
# <editor-fold desc="新KC-prepare_data-source">

data = 'nips34'  # 训练集 & 验证集
domain = 'source'
senario = 'conc'
file_out = [f'data/{data}/train.json', f'data/{data}/val.json']
shuffle = False
random_seed = 42
random.seed(random_seed)
drop_rate = 0.0

# domain类别
dict_domain = {'source': 0, 'target': 1}
cat_domain = dict_domain[domain]

# 获取数据集“三围”
with open(f'data/{data}/config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

# 读取数据（注意需要手动移动文件）
df = pd.read_csv(f'data/{data}/{domain}_{senario}.csv')

# 声明输出
list_formatted = []  # 训练集（无需处理，对错1：1）
list_validate = []  # 验证集（无需处理，对错1：1）

# 逐行（学生）遍历
for ind, row in df.iterrows():
    if (ind + 1) % 1000 == 0:
        print('{} of {}'.format(ind + 1, len(df)))

    # 提取当前遍历学生的信息（无需换算！！！！！）
    sid = int(row['uid'])
    pid = list(map(lambda x: int(x), row['questions'].split(',')))
    qid = [list(map(lambda x: int(x), kcs.split('_'))) for kcs in row['concepts'].split(',')]
    res = list(map(int, row['responses'].split(',')))
    n_log = len(res)

    # 判断序列长度是否达标，否则舍去
    if n_log < 15:
        continue

    """
    拆分训练、验证集（当前学生的所有记录都要）
        ——split_lists_by_index函数保证了训练和验证集学生完全一致
    """
    lists_t, lists_v = split_lists_by_index([pid, qid, res], fraction=13/15, seed=random_seed)
    pid_t, qid_t, res_t = lists_t
    pid_v, qid_v, res_v = lists_v

    # 对于符合要求的学生，遍历每一条交互数据，整合为所需的格式
    for _ind in range(len(pid_t)):    # 训练集
        __temp = qid_t[_ind]
        __temp.sort()
        temp_log = {  # 初始化当前学生的信息字典
            "user_id": sid,
            "exer_id": pid_t[_ind],   # int
            "score": res_t[_ind],     # int
            "domain": cat_domain,
            "knowledge_code": __temp  # list
        }
        # 将当前log条目添加至记录json列表中
        list_formatted.append(temp_log)

    for _ind in range(len(pid_v)):    # 验证集
        __temp = qid_v[_ind]
        __temp.sort()
        temp_log = {  # 初始化当前学生的信息字典
            "user_id": sid,
            "exer_id": pid_v[_ind],   # int
            "score": res_v[_ind],     # int
            "domain": cat_domain,
            "knowledge_code": __temp  # list
        }
        # 将当前log条目添加至记录json列表中
        if temp_log["score"] == 0:
            list_validate.append(temp_log)
        elif random.random() >= drop_rate:
            list_validate.append(temp_log)

# 检查
score_train = [elem["score"] for elem in list_formatted]
score_test = [elem["score"] for elem in list_validate]
score_train = sum(score_train) / len(score_train)
score_test = sum(score_test) / len(score_test)
print(f"correct rate: train set {score_train}, val set {score_test}")
print(f"size: train {len(list_formatted)}, val {len(list_validate)}")

# 乱序
if shuffle:
    random.shuffle(list_formatted)
    random.shuffle(list_validate)
# 保存
with open(file_out[0], 'w', encoding='utf-8') as json_file:
    json.dump(list_formatted, json_file, indent=4, ensure_ascii=False)
with open(file_out[1], 'w', encoding='utf-8') as json_file:
    json.dump(list_validate, json_file, indent=4, ensure_ascii=False)

# </editor-fold>

# <editor-fold desc="新KC-prepare_data-target">

data = 'nips34'  # 测试集
domain = 'target'
senario = 'conc'
file_out = [f'data/{data}/{senario}_target_train.json', f'data/{data}/{senario}_target_val.json', f'data/{data}/{senario}_test.json']
shuffle = False
random_seed = 42
random.seed(random_seed)
ratio_target = (0.5, 0.1, 0.4)  # 考虑到测试集需要人工平衡
drop_rate = 0.0

# domain类别
dict_domain = {'source': 0, 'target': 1}
cat_domain = dict_domain[domain]

# 获取数据集“三围”
with open(f'data/{data}/config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

# 读取数据
df = pd.read_csv(f'data/{data}/{domain}_{senario}.csv')

# 声明输出
list_train = []  # target训练集
list_val = []  # target验证集
list_test = []  # target测试集（均不需要进行人工平衡）

# 逐行（学生）遍历
for ind, row in df.iterrows():
    if (ind + 1) % 1000 == 0:
        print('{} of {}'.format(ind + 1, len(df)))

    # 提取当前遍历学生的信息（不需要换算！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！）
    sid = int(row['uid'])
    pid = list(map(lambda x: int(x), row['questions'].split(',')))
    qid = [list(map(lambda x: int(x), kcs.split('_'))) for kcs in row['concepts'].split(',')]
    res = list(map(int, row['responses'].split(',')))
    n_log = len(res)

    # 判断序列长度是否达标，否则舍去
    if n_log < 15:
        continue

    # 获取划分
    splits = split_list_by_ratio(list(range(n_log)), list(ratio_target))

    # 对于符合要求的学生，遍历每一条交互数据，整合为所需的格式
    for _ind in range(n_log):
        __temp = qid[_ind]
        __temp.sort()
        temp_log = {  # 初始化当前学生的信息字典
            "user_id": sid,
            "exer_id": pid[_ind],  # int
            "score": res[_ind],  # int
            "domain": cat_domain,
            "knowledge_code": __temp  # list
        }
        # 将当前log条目添加至记录json列表中
        if _ind in splits[0]:
            list_train.append(temp_log)
        elif _ind in splits[1]:
            list_val.append(temp_log)
        elif temp_log["score"] == 0:
            list_test.append(temp_log)
        elif random.random() >= drop_rate:
            """此处对测试集进行人工平衡"""
            list_test.append(temp_log)

# check
score_train = [elem["score"] for elem in list_train]
score_test = [elem["score"] for elem in list_test]
score_train = sum(score_train) / len(score_train)
score_test = sum(score_test) / len(score_test)
print(f"correct rate: train set {score_train}, test set {score_test}")
print(f"size: train {len(list_train)}, val {len(list_val)}, test {len(list_test)}")

# 乱序
if shuffle:
    random.shuffle(list_train)
    random.shuffle(list_val)
    random.shuffle(list_test)

# 保存
with open(file_out[0], 'w', encoding='utf-8') as json_file:
    json.dump(list_train, json_file, indent=4, ensure_ascii=False)
with open(file_out[1], 'w', encoding='utf-8') as json_file:
    json.dump(list_val, json_file, indent=4, ensure_ascii=False)
with open(file_out[2], 'w', encoding='utf-8') as json_file:
    json.dump(list_test, json_file, indent=4, ensure_ascii=False)

# </editor-fold>




# <editor-fold desc="实体图生成—prerequisite部分">

# 声明输出
list_prerequisite = []

for ind, row in df_KC_tree.iterrows():
    id_child = row['SubjectId']
    id_parent = row['ParentId']
    if not math.isnan(id_parent):
        list_prerequisite.append((int(id_parent), id_child))

# 完成遍历，去重
list_prerequisite = list(set(list_prerequisite))

# 写入txt文件
with open('data/nips34/NIPS34/K_Directed.txt', 'a') as file:
    for elem in list_prerequisite:
        file.write(str(elem[0]) + '\t' + str(elem[1]) + '\n')

# </editor-fold>


"""
暂时不利用这一点：相似性的表征需要进一步优化
    根据DCD的情况来看，不同深度的相似性显然不应该是一个东西，
    这里我们即使添加相似行，可能也应该限定再在叶子节点或者其父节点这些粒度较高的层级！！！
"""
# <editor-fold desc="实体图生成—similarity部分">

# 示例树形图创建
kc_tree = nx.DiGraph()
kc_tree.add_edges_from(list_prerequisite)
# 调用函数添加“相似”关系的无向边
# graph_with_similarity = add_similarity_edges(kc_tree)                     # 深度2
# graph_with_similarity = add_similarity_edges_depth1(kc_tree)              # 深度1
graph_with_similarity = add_similarity_edges_depth1_betweenLeaves(kc_tree)  # 仅限叶子节点之间存在相似
# 打印无向边
list_similarity = []
for edge in graph_with_similarity.edges(data=True):
    if edge[2].get('relation') == 'similar':
        list_similarity.append((edge[0], edge[1]))

# 写入txt文件
with open('data/nips34/NIPS34/K_Undirected.txt', 'a') as file:
    for elem in list_similarity:
        file.write(str(elem[0]) + '\t' + str(elem[1]) + '\n')

# </editor-fold>


"""
250113：Q-matrix 挂载优化，只挂载到对应的叶子节点上即可
250120：不再进行换算，以保证在ID squeeze时不会因为重复换算导致Qmat信息错误！！！—— 经检查无误，原来就没有换算
"""
# <editor-fold desc="实体图生成—Exer_Concept部分">

# prerequisite树形图创建
kc_tree = nx.DiGraph()
kc_tree.add_edges_from(list_prerequisite)
# 获取叶子节点
leaf_nodes = [node for node in kc_tree.nodes if kc_tree.out_degree(node) == 0]  # 316/388
# 声明输出
list_e2c = []
# 遍历Q-matrix，查找叶子节点（KC）及其对应的问题并记录
for ind, row in df_Qmat.iterrows():
    pid = row['QuestionId']
    qids = ast.literal_eval(row['SubjectId'])  # list
    for qid in qids:
        if qid in leaf_nodes:
            list_e2c.append((qid, pid))

# 写入txt文件
with open('data/nips34/NIPS34/Exer_Concept.txt', 'a') as file:
    for elem in list_e2c:
        file.write(str(elem[0]) + '\t' + str(elem[1]) + '\n')

# </editor-fold>


# <editor-fold desc="生成每个学生的history数据">

name_data = 'nips34'  # NIPS34
file_hist = f'data/{name_data}/history.json'

# 读取完整数据集
df_total = pd.read_csv(f'data/{name_data}/total.csv')
# 声明输出
dict_history = {}

# 逐行（学生）遍历
for ind, row in df_total.iterrows():
    if (ind+1) % 1000 == 0:
        print('{} of {}'.format(ind + 1, len(df_total)))

    # 提取当前遍历学生的信息
    sid = str(int(row['uid']))                                          # （换算统一移至align_index中进行）换算至从1起
    pid = list(map(lambda x: int(x), row['questions'].split(',')))      # history中的exer_id不进行换算
    # 写入
    dict_history[sid] = pid

# 保存
with open(file_hist, 'w', encoding='utf-8') as json_file:
    json.dump(dict_history, json_file, indent=4, ensure_ascii=False)

# </editor-fold>

