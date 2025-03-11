"""MOOC数据的预处理脚本"""
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import ast
import warnings
import copy
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略FutureWarning类型的警告
random.seed(42)  # 设置全局随机种子


# <editor-fold desc="读取数据">
name_data = 'MOOCRadar'
root_path = f'D:/Datasets/{name_data}'

with open(f'{root_path}/student-problem-coarse.json', 'r', encoding='utf-8') as file:
    df_coarse = json.load(file)
with open(f'{root_path}/student-problem-middle.json', 'r', encoding='utf-8') as file:
    df_middle = json.load(file)
# 读取文件并解析每行JSON
with open(f'{root_path}/problem.json', 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 去除每行的换行符并解析为字典
data = []
for line in lines:
    record = json.loads(line.strip())
    # 转换 `detail` 键的值
    if 'detail' in record:
        record['detail'] = ast.literal_eval(record['detail'])
    data.append(record)
# # 将数据写入新的JSON文件，格式化为列表
# with open(f'{root_path}/problem_fixed.json', 'w', encoding='utf-8') as outfile:
#     json.dump(data, outfile, indent=4, ensure_ascii=False)
# 重命名
df_exer = pd.DataFrame(data)

# </editor-fold>


# <editor-fold desc="拼合完整数据集">
n_stu = len(df_coarse)  # 14424
df = []
for sid, (e_c, e_m) in enumerate(zip(df_coarse, df_middle)):
    e_co = e_c['seq']  # list
    e_mi = e_m['seq']
    row = []
    for _ind in range(len(e_co)):
        if e_co[_ind]['log_id'] == e_mi[_ind]['log_id']:
            if e_co[_ind]['attempts'] == 1:
                temp = copy.deepcopy(e_co[_ind])
                temp['exercise_id'] = e_mi[_ind]['exercise_id']
                temp['user_id'] = sid
                row.append(temp)
            else:
                continue
        else:
            print('log id Dismatch!')
            raise ValueError
    df.extend(row)

print(len(df))  # 898756
df = pd.DataFrame(df)

# </editor-fold>


# <editor-fold desc="整合ID和题目文本信息并保存">

# 编码问题ID
n_exer = df_exer['problem_id'].value_counts()  # 9383（无重复）
n_exer = len(n_exer)
df_exer.reset_index(inplace=True)
df_exer.rename(columns={'index': 'exer_id'}, inplace=True)


# 将b, c, d三列的内容转为字符串后以'\n'拼接起来，作为content列
def concatenate_columns(_row):
    """自定义逐行操作函数"""
    cons = _row['concepts']
    cons = ', '.join(cons)
    opts = _row['detail']['option']
    if isinstance(opts, dict):
        opts = ', '.join([f'{_key}: {_val}' for _key, _val in opts.items()])
    else:
        opts = str(opts)
    ans = ast.literal_eval(_row['detail']['answer'])
    ans = ', '.join(ans)

    return ('[Title]\n' + _row['detail']['title'].strip() +
            '\n[Concepts]\n' + cons +
            '\n[Content]\n' + _row['detail']['content'].strip() +
            '\n[Options]\n' + opts +
            '\n[Answer]\n' + ans)


df_exer['content'] = df_exer.apply(concatenate_columns, axis=1)

result = df_exer.set_index('exer_id')['content'].to_dict()
# 保存
with open(f'{root_path}/problem_content.json', 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, indent=4, ensure_ascii=False)


# 应用自定义函数并将结果转换为一个字典
def convert_row_to_dict(_row):
    return {_row['problem_id']: {'exer_id': _row['exer_id'], 'content': _row['content']}}

dict_list = df_exer.apply(lambda _row: convert_row_to_dict(_row), axis=1)
result_dict = {k: v for d in dict_list for k, v in d.items()}
# 保存
with open(f'{root_path}/problem_fixed.json', 'w', encoding='utf-8') as json_file:
    json.dump(result_dict, json_file, indent=4, ensure_ascii=False)

# </editor-fold>


# <editor-fold desc="拼合信息，截断学生数，切分数据">

# 拼合信息
df['exer_id'] = df['problem_id'].apply(lambda x: result_dict[x]['exer_id'])

# 裁剪学生总数
n_clip = 2000
# 使用条件筛选方法删除A列大于2000的行
# df_filtered = df[df['user_id'] < n_clip]  # 89w -> 17.7w
# 随机抽取2000名学生
uid_list = random.sample(range(n_stu), n_clip)
df_filtered = df[df['user_id'].isin(uid_list)]  # ~11w

# 问题ID计数（题库9383，但record中出现过的仅有2513道）
print(len(df['exer_id'].value_counts()))        # 2513
pid_list = df_filtered['exer_id'].unique().tolist()
print(len(pid_list))                            # 1k7-1k8

# 切分source和target域
source, target = train_test_split(pid_list, test_size=0.2)  # , random_state=42
df_source = df_filtered[df_filtered['exer_id'].isin(source)]
df_target = df_filtered[df_filtered['exer_id'].isin(target)]
print(len(df_source))
print(len(df_target))
print(len(df_filtered))

# 在target中进一步切分val和test（source整个作为train）
df_val, df_test = train_test_split(df_target, test_size=0.2)  # , random_state=42
print(len(df_val))
print(len(df_test))

# </editor-fold>


# <editor-fold desc="格式转换，保存为NCDM标准输入json文件">
def convert_to_ncdm_format(data_df):
    ncdm_data = []
    # 获取所有唯一的用户ID和习题ID
    user_ids = data_df['user_id'].tolist()
    exer_ids = data_df['exer_id'].tolist()
    score = data_df['is_correct'].tolist()
    
    for _ind in range(len(user_ids)):
        ncdm_data.append(
            {
                "exer_id": exer_ids[_ind],
                "user_id": user_ids[_ind],
                "score": score[_ind],
                "knowledge_code": [0]
            }
        )
    return ncdm_data


# 转换训练、验证和测试数据集
train_data = convert_to_ncdm_format(df_source)
val_data = convert_to_ncdm_format(df_val)
test_data = convert_to_ncdm_format(df_test)

# 保存为JSON文件
with open(f'{root_path}/train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

with open(f'{root_path}/val.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, indent=4, ensure_ascii=False)

with open(f'{root_path}/test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

# </editor-fold>


# <editor-fold desc="文本信息处理，准备好分词和嵌入结果">
pass
# </editor-fold>
