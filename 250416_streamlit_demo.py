import streamlit as st
import pandas as pd
import glob
import os
import time
import json


# 直接读取KC名称信息为id2knowname
with open(r'data/id2knowname.json', 'r', encoding='utf-8') as file:
    id2knowname = json.load(file)

# 设置页面标题
st.title("精准学知识点学习路径可视化demo")

# 设置固定路径
FIXED_PATH = "data/results"  # 固定的本地路径，根据您的实际路径修改
n_files = 1000

def load_all_csv_files(directory_path, max_files=None):
    """读取目录下符合条件的CSV文件并合并为一个DataFrame

    参数:
        directory_path: 要读取的目录路径
        max_files: 最多读取的文件数量，默认为None表示读取所有符合条件的文件
    """
    try:
        all_files = glob.glob(os.path.join(directory_path, "*.csv"))
        if not all_files:
            st.error(f"在 {directory_path} 目录下没有找到CSV文件")
            return None

        # 筛选出大于50KB的文件
        valid_files = []
        for filename in all_files:
            file_size_kb = os.path.getsize(filename) / 1024
            if file_size_kb >= 50:
                valid_files.append(filename)

        if not valid_files:
            st.error(f"没有找到大于50KB的CSV文件")
            return None

        # 如果指定了max_files参数，则只取前max_files个文件
        if max_files is not None and max_files > 0:
            valid_files = valid_files[:max_files]
            st.info(f"根据设置，只读取前{max_files}个符合条件的文件")

        df_list = []
        for filename in valid_files:
            _df = pd.read_csv(filename)
            df_list.append(_df)

        # 合并所有数据
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")
        return None


# 加载数据
df = load_all_csv_files(FIXED_PATH, max_files=n_files)
if df is not None:
    st.success(f"成功加载 {len(df)} 行数据")

    # 计算tal_id的出现次数
    tal_id_counts = df['tal_id'].value_counts()
    # 筛选出现次数在500-1000之间的tal_id
    filtered_tal_ids = tal_id_counts[(tal_id_counts >= 200) & (tal_id_counts <= 500)].index.tolist()

    if not filtered_tal_ids:
        st.warning("没有value_count在200-500之间的tal_id")
    else:
        # 从数据中提取knowledge_id列的唯一值
        unique_c_values = df['knowledge_id'].unique()

        # 初始化图标状态字典（所有图标初始为未点亮）
        if 'icon_states' not in st.session_state:
            st.session_state.icon_states = {value.strip("jzx1.5_zsd_"): False for value in unique_c_values}
            for _val in unique_c_values:
                _key = _val.strip("jzx1.5_zsd_")
                if _key in id2knowname.keys():
                    st.session_state.icon_states[id2knowname[_key]] = False
                else:
                    st.session_state.icon_states[_val] = False
            st.session_state.lit_order = []  # 用于记录点亮顺序

        # 创建下拉选择框，让用户从筛选后的A列的唯一值中选择
        query_string = st.selectbox("请选择要查询的tal_id:", filtered_tal_ids)

        # 显示已点亮的图标
        st.subheader("知识点学习路径")
        if not st.session_state.lit_order:
            st.info("尚未学习任何知识点")
        else:
            # 创建一个网格布局来显示已点亮的图标
            num_cols = 5  # 每行显示5个图标
            rows = []
            current_row = []

            for i, icon_value in enumerate(st.session_state.lit_order):
                current_row.append(icon_value)
                if len(current_row) == num_cols or i == len(st.session_state.lit_order) - 1:
                    rows.append(current_row)
                    current_row = []

            for row in rows:
                cols = st.columns(num_cols)
                for i, icon_value in enumerate(row):
                    with cols[i]:
                        st.markdown(f"<h3 style='text-align: center; color: green;'>{icon_value}</h3>",
                                    unsafe_allow_html=True)

        if st.button("执行查询和点亮"):
            # 查询A列匹配的行
            matched_rows = df[df['tal_id'] == query_string]

            if len(matched_rows) == 0:
                st.warning(f"没有找到与 '{query_string}' 匹配的数据")
            else:
                # 按B列时间戳排序
                matched_rows['created_at'] = pd.to_datetime(matched_rows['created_at'])
                sorted_rows = matched_rows.sort_values('created_at')

                st.write(f"找到 {len(sorted_rows)} 行匹配数据，按时间排序如下:")
                st.dataframe(sorted_rows)

                # 记录上一次点亮的时间
                last_lit_time = None
                newly_lit = []  # 记录本次新点亮的图标

                # 逐行点亮图标
                count = 0
                for _, row in sorted_rows.iterrows():
                    c_value = row['knowledge_id']
                    c_value = c_value.strip("jzx1.5_zsd_")
                    if count < 10:
                        # 如果能查到精准学KC名称 & 图标未点亮，则点亮它
                        if c_value in id2knowname.keys():
                            c_value = id2knowname[c_value]

                            if not st.session_state.icon_states[c_value]:
                                count += 1

                                # 检查是否需要等待
                                current_time = time.time()
                                if last_lit_time and current_time - last_lit_time < 1.0:
                                    sleep_time = 1.0 - (current_time - last_lit_time)
                                    # st.write(f"等待 {sleep_time:.2f} 秒...")
                                    time.sleep(sleep_time)

                                # 点亮图标
                                st.session_state.icon_states[c_value] = True
                                st.session_state.lit_order.append(c_value)  # 添加到点亮顺序列表
                                newly_lit.append(c_value)
                                st.write(f"点亮图标: {c_value}")
                                last_lit_time = time.time()
                    else:
                        break

                # 更新图标显示
                if newly_lit:
                    st.rerun()  # 重新运行以更新UI

        # 添加重置按钮
        if st.button("重置所有图标"):
            st.session_state.icon_states = {value: False for value in unique_c_values}
            st.session_state.lit_order = []  # 清空点亮顺序列表
            st.rerun()
else:
    st.error("无法加载数据，请检查路径是否正确")

