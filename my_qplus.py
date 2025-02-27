import re
import requests
import json
import time
from typing import List, Optional, Union, Dict
import datetime


def convert_json_format(input_file, output_file):
    """
    python，将以下格式的json转为[{"system": <system_prompt>, "prompt": <user_prompt>}, ...]的格式：
        [
            {
                "instruction": "",
                "input": "[SYSTEM]\n<system_prompt>\n\n[USER]\n<user_prompt>",
                "output": "",
                "task_id": "",
                "raw_instruction": {
                "query": "",
                "answer": ""
                },
                "idx": 1
            },
            ...
        ]
    """
    # 读取输入JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换格式
    converted_data = []

    for item in data:
        idx = item['idx']
        input_text = item.get('input', '')

        # 使用正则表达式提取system和user部分
        system_match = re.search(r'\[SYSTEM\]\n(.*?)(?=\n\n\[USER\]|\Z)', input_text, re.DOTALL)
        user_match = re.search(r'\[USER\]\n(.*?)(?=\Z)', input_text, re.DOTALL)

        system_prompt = system_match.group(1).strip() if system_match else ""
        user_prompt = user_match.group(1).strip() if user_match else ""

        # 创建新格式
        converted_item = {
            "idx": idx,
            "system": system_prompt,
            "prompt": user_prompt
        }

        converted_data.append(converted_item)

    # 写入输出JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)

    return converted_data


class QwenPlusClient:
    def __init__(self, app_id: str = "1000080837", app_secret: str = "a2a2d65a243008e97f11593440abf48b"):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "http://apx-api.tal.com/v1"
        self.api_version = "2024-02-01"

    def _get_headers(self, priority: int = 9) -> Dict[str, str]:
        """Generate request headers with authentication"""
        api_key = f"{self.app_id}:{self.app_secret}"
        return {
            'Content-Type': 'application/json',
            'api-key': api_key,
            'x-apx-task-priority': str(priority)
        }

    def send_request(self, 
                     system: str,
                     prompt: str,
                     priority: int = 9,
                     temperature: float = 1.0,
                     max_new_tokens: int = 1024,
                     stop: Optional[List[str]] = None) -> Optional[str]:
        """Send a single request to Qwen Plus API"""
        url = f"{self.base_url}/async/chat?api-version={self.api_version}"
        
        if system:
            messages = [{"role": "system", "content": system}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": "qwen-plus",
            "messages": messages,
        }
        
        response = requests.post(
            url=url,
            headers=self._get_headers(priority),
            json=body
        )
        
        if response.ok:
            result = response.json()
            status = str(result['status'])
            return result['id'] if status != "4" else None
        return None
    
    def get_results(self, task_ids: Union[str, List[str]]) -> Dict:
        """Get results for one or multiple task IDs"""
        def _post_result_url(task_ids):
            url = f"{self.base_url}/async/results/detail"
            response = requests.post(
                url=url,
                headers=self._get_headers(),
                json={"task_ids": task_ids}
            )
            res_dict = response.json()
            return res_dict

        if isinstance(task_ids, str):
            task_ids = [task_ids]

        res_dict = _post_result_url(task_ids)
        return_res_dict = []

        try:
            for task_id, task_info in res_dict["data"].items():
                if task_info.get('status') == 3:
                    try:
                        return_res_dict.append({
                            'task_id': task_id,
                            'output': task_info['response']['choices'][0]['message']['content']
                        })
                    except (KeyError, IndexError) as e:
                        print(f"Error processing task {task_id}: {e}")
                        continue
        except Exception as e:
            print(f"Error getting results: {e}")
            return None

        # Sort results to match input order
        return_res_dict = sorted(return_res_dict, key=lambda x: task_ids.index(x['task_id']))
        return return_res_dict


if __name__ == "__main__":

    # task_name = 'sct_all_coll_user_'
    # task_name = 'sct_all_coll_exer_'
    task_name = 'sct_all_diag_exer_20Clip_'
    # task_name = 'sct_all_diag_user_20Clip_'
    ind_start = 500
    ind_end = 1000

    raw_input = f'KCD/output_{task_name}.json'
    formatted_input = f'KCD/input_{task_name}_{ind_start}_{ind_end}.json'
    response_json = f'KCD/response_{task_name}_{ind_start}_{ind_end}.json'

    with open(response_json, 'w', encoding='utf-8') as file:
        json.dump([], file, ensure_ascii=False, indent=4)

    """QwenPlusClient"""
    client = QwenPlusClient()
    prompts = convert_json_format(raw_input, formatted_input)

    # print(len(prompts))
    print(task_name)
    print(ind_start)
    print(ind_end)
    # raise ValueError

    print(min(ind_end,len(prompts)-1))
    prompts = prompts[ind_start: min(ind_end,len(prompts)-1)]
    
    count = 0
    new_data = []
    for prompt in prompts:
        idx = prompt['idx']
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"), f', start processing prompt idx {idx}')
        # 发送请求
        task_id = client.send_request(system=prompt['system'], prompt=prompt['prompt'])
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"), f', request sent, task_id: {task_id}')
        # 接收响应
        while True:
            res = client.get_results(task_id)
            if res:
                # print(res)
                if len(res) == 1:
                    res = res[0]
                    res['idx'] = idx      # 将index信息整合至响应字典
                    new_data.append(res)  # 临时记录至new_data列表
                    count += 1            # 计数+1
                    # 每收集100条响应，读写更新一次响应记录文件
                    if count % 100 == 0:
                        with open(response_json, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                        data.extend(new_data)
                        with open(response_json, 'w', encoding='utf-8') as file:
                            json.dump(data, file, ensure_ascii=False, indent=4)
                        new_data = []     # 重置临时记录列表
                # 输出成功获取响应的时间，并跳出while循环，处理下一prompt
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"), f', get response.')
                break
            time.sleep(2)

    # 跳出主循环（成功获取所有prompt的响应）后，写入最后的不足100条响应
    with open(response_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data.extend(new_data)
    with open(response_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print('finish!')
