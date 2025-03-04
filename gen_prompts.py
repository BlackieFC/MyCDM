import pandas as pd
import json
import ast
import os
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略FutureWarning类型的警告
import random
random.seed(42)  # 全局随机种子设置


# <editor-fold desc="自定义函数">


def extract_substring(text):
    """
    截取字符串字串：从第一个\"summarization\":之后起，直至第一个\"reasoning\":之前
    """
    match = re.search(r'"summarization":(.*?)"reasoning":', text, re.DOTALL)
    if match:
        return match.group(1).strip().strip(',').strip('"')
    return None


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created missing directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def keep_latex_format(latex_list):
    """将列表转换为字符串表示，同时保留LaTeX表示法的转义字符"""
    return latex_list.replace('\\\\', '\\')


def get_qmat(path_in):
    """获取q矩阵"""
    # 读取CSV文件
    _df = pd.read_csv(path_in)

    # 初始化一个字典用于存储结果
    result_dict = {}
    for _ind, _row in _df.iterrows():
        result_dict[_row['item_id']] = ast.literal_eval(_row['knowledge_code'])

    # 获取item_id列的集合
    item_ids = _df['item_id'].unique()
    # 处理user_ids
    item_ids = list(item_ids)
    item_ids.sort()

    # 返回结果列表
    return result_dict, item_ids


def get_user_prompt(path_in, q_names, p_contents, max_len=None):
    """
    请给出python代码，用于读取并处理内容如下形式的csv文件，要求首先获取user_id列的集合A，再遍历A中的每一个user_id，
    将csv文件中匹配的所有行以{item_id:内容,score:内容}形式的字典作为元素组成列表B，最后将键值对user_id:B添加至字典C中，
    遍历结束后返回列表C：
        user_id,item_id,score
        22221,1998,0.0
        6356,2469,0.0
        1501,1970,1.0
        7258,290,1.0
        ...
    """
    # 初始化一个字典用于存储结果
    result_dict = {}

    # 读取CSV文件
    with open(path_in, 'r', encoding='utf-8') as fi:
        _recs = json.load(fi)  # list of dicts

    # 拆分出每一个学生的交互序列
    _set_uid = {}
    # 保序遍历
    for d in _recs:
        _key = d['user_id']
        # 如果结果字典中不存在该键，则创建一个新的键值对
        if _key not in _set_uid:
            _set_uid[_key] = []
        # 将当前字典加入到对应的键的列表中
        _set_uid[_key].append(d)

    # 遍历user_ids集合中的每一个user_id
    for user_id, rec_seq in _set_uid.items():
        # 直接将json转为df方便操作
        rec_seq = pd.DataFrame(rec_seq)
        # 将每一行转换成字典形式并加入列表
        user_items = rec_seq.apply(
            lambda x: {'content': p_contents[str(x['exer_id'])],
                       'concept': '; '.join(q_names.loc[q_names['SubjectId'].isin(x['knowledge_code']), 'Name'].tolist()),
                       'answer': True if x['score'] else False
                       },
            axis=1
        ).tolist()
        # 最大长度截断
        if max_len is not None and max_len < len(user_items):
            user_items = user_items[:max_len]
        # 记录当前学生的交互序列
        result_dict[user_id] = user_items

    # 返回结果列表
    return result_dict


def get_user_diagnosis(path_in, q_names, p_contents, u_sum, p_sum, max_len=None):
    """
    诊断用：请给出python代码，用于读取并处理内容如下形式的csv文件，要求首先获取user_id列的集合A，再遍历A中的每一个user_id，
    将csv文件中匹配的所有行以{item_id:内容,score:内容}形式的字典作为元素组成列表B，最后将键值对user_id:B添加至字典C中，
    遍历结束后返回列表C：
        user_id,item_id,score
        22221,1998,0.0
        6356,2469,0.0
        1501,1970,1.0
        7258,290,1.0
        ...
    """
    # 初始化一个字典用于存储结果
    result_dict = {}

    # 读取CSV文件
    with open(path_in, 'r', encoding='utf-8') as fi:
        _recs = json.load(fi)  # list of dicts

    # 拆分出每一个学生的交互序列
    _set_uid = {}
    # 保序遍历
    for d in _recs:
        _key = d['user_id']
        # 如果结果字典中不存在该键，则创建一个新的键值对
        if _key not in _set_uid:
            _set_uid[_key] = []
        # 将当前字典加入到对应的键的列表中
        _set_uid[_key].append(d)

    # 遍历user_ids集合中的每一个user_id
    for uid, rec_seq in _set_uid.items():
        # 直接将json转为df方便操作
        rec_seq = pd.DataFrame(rec_seq)
        # 将每一行转换成字典形式并加入列表
        user_items = rec_seq.apply(
            lambda x: {'content': p_contents[str(x['exer_id'])],
                       'concept': '; '.join(q_names.loc[q_names['SubjectId'].isin(x['knowledge_code']), 'Name'].tolist()),
                       'answer': True if x['score'] else False,
                       'exercise_profile': p_sum[int(x['exer_id'])]
                       },
            axis=1
        ).tolist()
        # 最大长度截断
        if max_len is not None and max_len < len(user_items):
            user_items = user_items[:max_len]
        # 记录当前学生的交互序列
        result_dict[uid] = user_items

    # 返回结果列表
    return result_dict


def get_exer_prompt(path_in, q_names, p_contents, max_len=None):
    """用于返回所有题目的user prompt信息"""
    # 初始化一个字典用于存储结果
    result_dict = {}

    # 读取CSV文件
    with open(path_in, 'r', encoding='utf-8') as fi:
        _recs = json.load(fi)  # list of dicts

    # 拆分出每一个题目的所有交互
    _set_pid = {}
    # 保序遍历
    for d in _recs:
        _key = d['exer_id']
        # 如果结果字典中不存在该键，则创建一个新的键值对
        if _key not in _set_pid:
            _set_pid[_key] = []
        # 将当前字典加入到对应的键的列表中
        _set_pid[_key].append(d)

    # 遍历_set_pid集合中的每一个exer_id
    for _exer_id, rec_seq in _set_pid.items():
        # 直接将json转为df方便操作
        rec_seq = pd.DataFrame(rec_seq)
        # 将每一行转换成字典形式并加入列表
        _history = rec_seq.apply(lambda x: {'answer': True if x['score'] else False}, axis=1).tolist()
        # 最大长度截断
        if max_len is not None and max_len < len(_history):
            _history = _history[:max_len]

        # 记录当前学生的交互序列
        result_dict[_exer_id] = {
            'exercise_content': p_contents[str(_exer_id)],
            'related_concept': '; '.join(q_names.loc[q_names['SubjectId'].isin(rec_seq.loc[0, 'knowledge_code']), 'Name'].tolist()),
            'history': _history
        }

    # 返回结果列表
    return result_dict


def get_exer_diagnosis(path_in, q_names, p_contents, u_sum, p_sum, max_len=None):
    """诊断用：用于返回所有题目的user prompt信息"""
    # 初始化一个字典用于存储结果
    result_dict = {}

    # 读取CSV文件
    with open(path_in, 'r', encoding='utf-8') as fi:
        _recs = json.load(fi)  # list of dicts

    # 拆分出每一个题目的所有交互
    _set_pid = {}
    # 保序遍历
    for d in _recs:
        _key = d['exer_id']
        # 如果结果字典中不存在该键，则创建一个新的键值对
        if _key not in _set_pid:
            _set_pid[_key] = []
        # 将当前字典加入到对应的键的列表中
        _set_pid[_key].append(d)

    # 遍历_set_pid集合中的每一个exer_id
    for _exer_id, rec_seq in _set_pid.items():
        # 直接将json转为df方便操作
        rec_seq = pd.DataFrame(rec_seq)
        # 将每一行转换成字典形式并加入列表
        _history = rec_seq.apply(lambda x: {'answer': True if x['score'] else False,
                                            'student_profile': u_sum[x['user_id']]
                                            },
                                 axis=1
                                 ).tolist()
        # 最大长度截断
        if max_len is not None and max_len < len(_history):
            _history = _history[:max_len]

        # 记录当前学生的交互序列
        result_dict[_exer_id] = {
            'exercise_profile': p_sum[int(_exer_id)],
            'exercise_content': p_contents[str(_exer_id)],
            'related_concept': '; '.join(q_names.loc[q_names['SubjectId'].isin(rec_seq.loc[0, 'knowledge_code']), 'Name'].tolist()),
            'history': _history
        }

    # 返回结果列表
    return result_dict


# </editor-fold>


# <editor-fold desc="（视情况）读取NIPS34原始文件">
"""读取数据，统计数量"""
df = pd.read_csv(r'C:\Games\NIPS\data\train_data\train_task_3_4.csv',
                 usecols=['QuestionId', 'UserId', 'IsCorrect'])
df_Qmat = pd.read_csv(r'C:\Games\NIPS\data\metadata\question_metadata_task_3_4.csv')
df = df.merge(df_Qmat[['QuestionId', 'SubjectId']], on='QuestionId', how='left')  # 共约138w条交互
df_KC_tree = pd.read_csv(r'C:\Games\NIPS\data\metadata\subject_metadata.csv')
with open(f'data/nips34_short.json', 'r', encoding='utf-8') as file:
    exer_content = json.load(file)

# 获取学生ID
n_stu = df['UserId'].value_counts()       # 4918
sids = [sid for sid, _ in n_stu.items()]  # 1-6147，共4918人（非致密）
sids.sort()
n_stu = len(sids)

# 获取问题ID
n_exer = df['QuestionId'].value_counts()   # 948
pids = [pid for pid, _ in n_exer.items()]  # 0-947，共948题（致密）
pids.sort()
n_exer = len(pids)

# 获取知识点ID
kids = df_KC_tree['SubjectId'].tolist()    # 388
kids.sort()
n_conc = len(kids)                         # 3-1988，共388个KC（非致密）

# check
result = set()
for ind, row in df_Qmat.iterrows():        # 3-655，题库涉及的KC（实际操作时将根部的两层KC剔除）
    temp = ast.literal_eval(row['SubjectId'])
    result.update(temp)
print(min(result))
print(max(result))

# </editor-fold>


# <editor-fold desc="（已完成）全量数据集划分">

# cols = ['user_id', 'exer_id', 'knowledge_code', 'score']
# json_result = []
# for ind, row in df.iterrows():
#     temp_log = {
#         "user_id": row['UserId'],
#         "exer_id": row['QuestionId'],
#         "score": row['IsCorrect'],
#         "knowledge_code": ast.literal_eval(row['SubjectId'])
#     }
#     json_result.append(temp_log)
#
# with open('nips34_all.json', 'w', encoding='utf-8') as json_file:
#     json.dump(json_result, json_file, indent=4, ensure_ascii=False)
#
# from sklearn.model_selection import train_test_split
#
# train_set, temp = train_test_split(json_result, test_size=0.3, random_state=42)
# validation_set, test_set = train_test_split(temp, test_size=2/3, random_state=42)
#
# with open('nips34_train.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_set, json_file, indent=4, ensure_ascii=False)
# with open('nips34_val.json', 'w', encoding='utf-8') as json_file:
#     json.dump(validation_set, json_file, indent=4, ensure_ascii=False)
# with open('nips34_test.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_set, json_file, indent=4, ensure_ascii=False)

# </editor-fold>


# <editor-fold desc="输入参数">

"""最大长度截断"""
maxlen = 20      # 默认值为None，不做截断
senario = 'longtail'  # all，只使用于all情景（longtail情景仅训练集为真子集）

"""输入数据集位置"""
doc_data = f'data/NIPS34/{senario}/'
file_names = ['train']  # 'train'，生成Embedding只需要使用训练集

"""prompt文件路径"""
# out_data = 'coll_info_collection/user_system_prompt'
# out_data = 'coll_info_collection/exer_system_prompt'
# out_data = 'diagnosis_generation/user_system_prompt'
out_data = 'diagnosis_generation/exer_system_prompt'
file_sys_prompt = f'D:/PythonProjects/KCD/LLM_diagnosis/{out_data}.txt'

"""输出文件"""
# 实验名
# exp_name = 'coll_user'
# exp_name = 'coll_exer'
# exp_name = 'diag_user'
exp_name = 'diag_exer'
task_curr = f"sct_{senario}_{exp_name}_"  # 指定任务名(call_gpt4时的task名)：姓名_情景_步骤_对象
# 路径
out_data = doc_data + out_data
ensure_path_exists(out_data)

# </editor-fold>


# <editor-fold desc="(已运行)人造长尾数据集">

# file_path = f'{doc_data}train.json'
# file_out = f'{doc_data}train_longtail.json'
#
# with open(file_path, 'r', encoding='utf-8') as fi:
#     recs = json.load(fi)  # list of dicts
#
# recs = pd.DataFrame(recs)
# exer_count = recs['exer_id'].value_counts()
#
# result = exer_count[exer_count <= 3]
# result = list(set(result) | set(random.sample(list(range(948)), 948//3)))
#
# exer_count = {e:0 for e in result}
# list_filtered = []
# for ind, row in recs.iterrows():
#     if row['exer_id'] in exer_count.keys():
#         # 需要进行人为处理的pid
#         if exer_count[row['exer_id']] < 3:
#             exer_count[row['exer_id']] += 1
#             list_filtered.append(row.to_dict())
#     else:
#         # 无需进行人为处理的pid，直接转存
#         list_filtered.append(row.to_dict())
#
# # 将人工控制后的数据保存为json
# with open(file_out, 'w', encoding='utf-8') as json_file:
#     json.dump(list_filtered, json_file, indent=4, ensure_ascii=False)
#
# # check
# list_filtered = pd.DataFrame(list_filtered)
# print(len(recs))
# print(len(list_filtered))
# exer_count = list_filtered['exer_id'].value_counts()
# print(len(exer_count[exer_count <= 3]))

# </editor-fold>


# <editor-fold desc="process coll info">

exer_sum = {}
user_sum = {}

if exp_name == 'diag_user' or exp_name == 'diag_exer':
    with open('data/NIPS34/all/coll_info_collection/sct_all_coll_exer_20Clip_367987_all4gpt_res.json', 'r', encoding='utf-8') as file:
        exer_coll_info = json.load(file)
    with open('data/NIPS34/all/coll_info_collection/sct_all_coll_user_20Clip_367987_all4gpt_res.json', 'r', encoding='utf-8') as file:
        user_coll_info = json.load(file)

    count = 0
    for elem in exer_coll_info:
        key = elem['idx']
        val = extract_substring(elem['output'])
        exer_sum[key] = val
        if val is None:
            count += 1
    print(count)

    count = 0
    for elem in user_coll_info:
        key = elem['idx']
        val = extract_substring(elem['output'])
        user_sum[key] = val
        if val is None:
            count += 1
    print(count)

# </editor-fold>


# <editor-fold desc="coll_user / diag_user">

if exp_name == 'coll_user' or exp_name == 'diag_user':

    # 初始化-调用接口的标准传入格式
    data_prompt = {
        "instruction": "",
        "input": "",
        "output": "",
        "task_id": "",
        "raw_instruction": {
            "query": "",
            "answer": ""
        }
    }
    # 读取-system prompt
    sys_prompt = ""
    with open(file_sys_prompt, 'r') as f:
        for line in f.readlines():
            sys_prompt += line
    # 读取-待处理文件
    user_prompts = []
    for file_name in file_names:  # 实际操作时，只有train.py
        file_path = f'{doc_data}{file_name}.json'

        if exp_name == 'coll_user':
            user_prompts.append(get_user_prompt(file_path, q_names=df_KC_tree, p_contents=exer_content, max_len=maxlen))
        else:
            user_prompts.append(get_user_diagnosis(file_path,
                                                   q_names=df_KC_tree,
                                                   p_contents=exer_content,
                                                   u_sum=user_sum,
                                                   p_sum=exer_sum,
                                                   max_len=maxlen)
                                )

    # 获取完整prompt
    for file_name, user_prompt in zip(file_names, user_prompts):
        uid_all = list(user_prompt.keys())
        uid_all.sort()
        _prompts = []
        for user_id in uid_all:
            data = data_prompt.copy()
            if exp_name == 'coll_user':
                data['input'] = sys_prompt + 'STUDY HISTORY: ' + json.dumps(user_prompt[user_id], ensure_ascii=False)
            else:
                data['input'] = sys_prompt + 'BASIC INFORMATION: ' + user_sum[user_id] + '\nSTUDY HISTORY: ' + json.dumps(user_prompt[user_id], ensure_ascii=False)
            data['idx'] = int(user_id)
            _prompts.append(data)

        # 保存为json文件
        json_outputs = json.dumps(_prompts, ensure_ascii=False, indent=5, separators=(',', ': '))
        with open(f'{out_data}/output_{task_curr}.json', 'w', encoding='utf-8') as json_file:
            json_file.write(json_outputs)

# </editor-fold>


# <editor-fold desc="coll_exer / diag_exer">

if exp_name == 'coll_exer' or exp_name == 'diag_exer':
    # 初始化-调用接口的标准传入格式
    data_prompt = {
        "instruction": "",
        "input": "",
        "output": "",
        "task_id": "",
        "raw_instruction": {
            "query": "",
            "answer": ""
        }
    }
    # 读取-system prompt
    sys_prompt = ""
    with open(file_sys_prompt, 'r') as f:
        for line in f.readlines():
            sys_prompt += line
    # 读取-待处理文件
    user_prompts = []
    for file_name in file_names:
        file_path = f'{doc_data}{file_name}.json'
        if exp_name == 'coll_exer':
            user_prompts.append(get_exer_prompt(file_path, q_names=df_KC_tree, p_contents=exer_content, max_len=maxlen))
        else:
            user_prompts.append(
                get_exer_diagnosis(file_path,
                                   q_names=df_KC_tree,
                                   p_contents=exer_content,
                                   u_sum=user_sum,
                                   p_sum=exer_sum,
                                   max_len=maxlen)
            )

    # 获取完整prompt
    for file_name, user_prompt in zip(file_names, user_prompts):
        pid_all = list(user_prompt.keys())
        pid_all.sort()

        _prompts = []
        for exer_id in pid_all:
            data = data_prompt.copy()
            dict_exer = user_prompt[exer_id]
            history = dict_exer.pop('history')  # 将history提取出来
            data['input'] = sys_prompt + 'BASIC INFORMATION: ' + json.dumps(dict_exer, ensure_ascii=False, indent=4) + '\nSTUDY HISTORY: ' + json.dumps(history, ensure_ascii=False)
            data['idx'] = int(exer_id)
            _prompts.append(data)

        # 保存为json文件
        json_outputs = json.dumps(_prompts, ensure_ascii=False, indent=5, separators=(',', ': '))
        with open(f'{out_data}/output_{task_curr}.json', 'w', encoding='utf-8') as json_file:
            json_file.write(json_outputs)

# </editor-fold>
