import streamlit as st
import pandas as pd
import glob
import os
import time
import json


# 使用缓存加载JSON数据
@st.cache_data
def load_id2knowname():
    with open(r'data/id2knowname.json', 'r', encoding='utf-8') as file:
        return json.load(file)

# 设置页面标题
st.title("精准学知识点学习路径可视化demo")

# 设置固定路径
FIXED_PATH = "data/results"  # 固定的本地路径，根据您的实际路径修改
n_files = 1000

# 使用缓存进行文件读取
@st.cache_data
def load_all_csv_files(directory_path, max_files=None):
    """读取目录下符合条件的CSV文件并合并为一个DataFrame"""
    try:
        # 获取所有CSV文件并按大小排序
        all_files = glob.glob(os.path.join(directory_path, "*.csv"))
        if not all_files:
            return None

        # 筛选大于50KB的文件，并计算文件大小一次
        valid_files_with_size = [(f, os.path.getsize(f)/1024) for f in all_files]
        valid_files = [f for f, size in valid_files_with_size if size >= 50]

        if not valid_files:
            return None

        # 如果指定了max_files，只读取前max_files个文件
        if max_files is not None and max_files > 0:
            valid_files = valid_files[:max_files]

        # 只读取必要的列，减少内存占用
        necessary_columns = ['tal_id', 'knowledge_id', 'created_at']
        
        # 使用pandas的优化选项批量读取
        combined_df = pd.concat([
            pd.read_csv(f, usecols=necessary_columns, dtype={
                'tal_id': str,
                'knowledge_id': str
            }) for f in valid_files
        ], ignore_index=True)
        
        return combined_df
    except Exception as e:
        return None


# 加载数据
id2knowname = load_id2knowname()

# 显示加载中的状态信息
with st.spinner("正在加载数据..."):
    df = load_all_csv_files(FIXED_PATH, max_files=n_files)

if df is not None:
    st.success(f"成功加载 {len(df)} 行数据")
    
    # 预处理数据以提高性能
    # 1. 计算tal_id的出现次数
    @st.cache_data
    def get_filtered_tal_ids(dataframe):
        tal_id_counts = dataframe['tal_id'].value_counts()
        return tal_id_counts[(tal_id_counts >= 200) & (tal_id_counts <= 500)].index.tolist()
    
    filtered_tal_ids = get_filtered_tal_ids(df)

    if not filtered_tal_ids:
        st.warning("没有value_count在200-500之间的tal_id")
    else:
        # 2. 预处理knowledge_id (只做一次)
        @st.cache_data
        def get_unique_knowledge(dataframe):
            return dataframe['knowledge_id'].unique()
        
        unique_c_values = get_unique_knowledge(df)

        # 初始化session状态（仅在首次运行时）
        if 'icon_states' not in st.session_state:
            # 使用字典推导式简化处理
            st.session_state.icon_states = {}
            for _val in unique_c_values:
                _key = _val.strip("jzx1.5_zsd_")
                display_val = id2knowname.get(_key, _val)
                st.session_state.icon_states[display_val] = False
            
            st.session_state.lit_order = []

        # 创建下拉选择框
        query_string = st.selectbox("请选择要查询的tal_id:", filtered_tal_ids)

        # 显示已点亮的图标
        st.subheader("知识点学习路径")
        if not st.session_state.lit_order:
            st.info("尚未学习任何知识点")
        else:
            # 优化显示逻辑，减少列计算次数
            max_icons_per_row = 5
            
            # 创建行列布局
            total_icons = len(st.session_state.lit_order)
            rows_needed = (total_icons + max_icons_per_row - 1) // max_icons_per_row
            
            for row_idx in range(rows_needed):
                cols = st.columns(max_icons_per_row)
                start_idx = row_idx * max_icons_per_row
                end_idx = min(start_idx + max_icons_per_row, total_icons)
                
                for col_idx, icon_idx in enumerate(range(start_idx, end_idx)):
                    icon_value = st.session_state.lit_order[icon_idx]
                    cols[col_idx].markdown(f"<h3 style='text-align: center; color: green;'>{icon_value}</h3>",
                                   unsafe_allow_html=True)

        if st.button("执行查询和点亮"):
            # 查询对应行
            matched_rows = df[df['tal_id'] == query_string].copy()  # 使用.copy()避免SettingWithCopyWarning

            if len(matched_rows) == 0:
                st.warning(f"没有找到与 '{query_string}' 匹配的数据")
            else:
                # 一次性转换时间戳，然后排序
                matched_rows['created_at'] = pd.to_datetime(matched_rows['created_at'])
                sorted_rows = matched_rows.sort_values('created_at')

                st.write(f"找到 {len(sorted_rows)} 行匹配数据，按时间排序如下:")
                st.dataframe(sorted_rows)

                # 批量处理点亮操作，减少循环和时间检查次数
                newly_lit = []
                count = 0
                
                # 预处理所有行以提高效率
                knowledge_list = []
                for _, row in sorted_rows.iterrows():
                    c_value = row['knowledge_id'].strip("jzx1.5_zsd_")
                    display_value = id2knowname.get(c_value, c_value)
                    knowledge_list.append(display_value)
                
                # 批量点亮处理
                for display_value in knowledge_list:
                    if count >= 10:
                        break
                        
                    if not st.session_state.icon_states.get(display_value, False):
                        st.session_state.icon_states[display_value] = True
                        st.session_state.lit_order.append(display_value)
                        newly_lit.append(display_value)
                        count += 1
                        
                        # 添加200ms延迟来模拟点亮动画效果，而不是1秒
                        time.sleep(0.2)
                
                if newly_lit:
                    st.write(f"新点亮了{len(newly_lit)}个知识点")
                    st.rerun()

        # 添加重置按钮
        if st.button("重置所有知识点"):
            for key in st.session_state.icon_states:
                st.session_state.icon_states[key] = False
            st.session_state.lit_order = []
            st.rerun()
else:
    st.error("无法加载数据，请检查路径是否正确")
