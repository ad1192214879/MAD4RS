import json
# from vllm import LLM, SamplingParams

import json
import logging
from vllm import LLM, SamplingParams
from datetime import datetime
import pytz
from typing import Dict, List


Clean_User_Advocate_Initial_Judgment_Prompt="""Your role: Clean User Advocate. Your objective is to analyze the given user interaction sequence and explain why it appears clean, supporting the claim that the user is a normal, non-malicious user.
Scenario: Art Recommendation Scenario
User ID: {user_id}
Interaction Sequence:
 (The format for each line is <Item ID>: <Item Description>)
{interaction_sequence}
Your Output format:
Stance: The user <User ID> is a normal user, and their interaction sequence is clean.
Confidence: High.
Explanation: <Provide a clear reasoning process, highlighting behavioral consistency, engagement patterns, and other relevant factors that indicate normal user activity.>"""

Noisy_User_Analyst_Initial_Judgment_Prompt="""Your Role: Noisy User Analyst. Your objective is to analyze the given user interaction sequence, identify noisy items, and explain why the user is a normal user despite the presence of noise.
Scenario: Art Recommendation Scenario
User ID: {user_id}
Interaction Sequence:
 (The format for each line is <Item ID>: <Item Description>)
{interaction_sequence}
Output Format:
Stance: The user <User ID> is a normal user, but the sequence contains noise items.
Confidence: High.
Noisy Items: <List of identified noisy item IDs>.
Explanation: <Provide a clear reasoning process, discussing patterns, inconsistencies, and possible reasons for noise in the interaction sequence.>"""

Fraud_Investigator_Initial_Judgment_Prompt="""Your role: Fraud Investigator. Your objective is to analyze the given user interaction sequence, identify signs of fabrication, and explain why the user is a fraudster who has manipulated the sequence to include a promotional product.
Scenario: art recommendation scenario
User ID: {user_id}
Interaction Sequence:
 (The format for each line is <Item ID>: <Item Description>)
{interaction_sequence}
Your Output format:
Stance: The user <User ID> is a fraudster, and the interaction sequence is fabricated by an attacker to include a promotional product.
Confidence: High.
Explanation: <Provide a clear reasoning process, highlighting suspicious patterns, anomalies, and indicators of fraudulent behavior in the sequence.>"""


Challenge_Generation_Prompt="""Now it is the debate stage. Your role is {role_A}, and you need to question the opinions of {role_B}.
This is the content of {role_B}'s opinions:
{role_B_content}
The content of your question is:"""

Clean_User_Advocate_Defend_Generation_Prompt="""Now it is the debate stage. Your role is Clean_User_Advocate. You need to respond to the doubts raised by {role_B} about your views, and adjust your confidence in your views according to the rationality of the doubts. There are three levels of confidence: High, Medium, and Low.
{role_B}'s doubts about you:
{Content_of_doubt}
Output Format:
Stance: The user <User ID> is a normal user, and their interaction sequence is clean.
Confidence: <Adjusted confidence>.
Response: <Response Content>."""

Noisy_User_Analyst_Defend_Generation_Prompt="""Now it is the debate stage. Your role is Noisy_User_Analyst. You need to respond to the doubts raised by {role_B} about your views, and adjust your confidence in your views according to the rationality of the doubts. There are three levels of confidence: High, Medium, and Low.
{role_B}'s doubts about you:
{Content_of_doubt}
Output Format:
Stance: The user <User ID> is a normal user, but the sequence contains noise items.
Confidence: <Adjusted confidence>.
Noisy Items: <List of identified noisy item IDs>.
Response: <Response Content>."""

Fraud_Investigator_Defend_Generation_Prompt="""Now it is the debate stage. Your role is Fraud_Investigator. You need to respond to the doubts raised by {role_B} about your views, and adjust your confidence in your views according to the rationality of the doubts. There are three levels of confidence: High, Medium, and Low.
{role_B}'s doubts about you:
{Content_of_doubt}
Output Format:
Stance: The user <User ID> is a fraudster, and the interaction sequence is fabricated by an attacker to include a promotional product.
Confidence: <Adjusted confidence>.
Response: <Response Content>."""


Self_Refine_Prompt="""After several rounds of debate, you can reflect on the final explanation and adjust the confidence. There are three levels of confidence: High, Medium, and Low.
Output Format:
Stance: The user <User ID> is a normal user, but the sequence contains noise items.
Confidence: <Adjusted confidence>.
Explanation: <Explanation content>."""

Noisy_Self_Refine_Prompt="""After several rounds of debate, you can reflect on the final explanation and adjust the confidence. There are three levels of confidence: High, Medium, and Low.
Output Format:
Stance: The user <User ID> is a normal user, but the sequence contains noise items.
Confidence: <Adjusted confidence>.
Noisy Items: <List of identified noisy item IDs>.
Explanation: <Explanation content>."""


Final_Judge_Prompt="""You are a judge. The three characters have analyzed and judged based on the user's historical interaction sequence. You decide which of their three mutually exclusive opinions is correct and output them in the specified format.
User historical interaction sequence:
{interaction_sequence}
Clean User Advocate:
{clean_content}
Noisy User Analyst:
{noisy_content}
Fraud Investigator:
{fraduster_content}
Output Format:
Role: <correct role>.
Stance: <Stance content>.
Confidence: <confidence>.
Noisy Items: <List of identified noisy item IDs if exist>.
Explanation: <Explanation content>."""


def get_interactions_with_text():
    # 加载 user_dict.json 和 item_dict.json
    with open("/home/aiding/LoRec/data/arts1000dense/user_dict.json", "r", encoding="utf-8") as f:
        user_dict = json.load(f)

    with open("/home/aiding/LoRec/data/arts1000dense/item_dict.json", "r", encoding="utf-8") as f:
        item_dict = json.load(f)

    # 构建新的结构
    output_dict = {}
    for user_id, item_list in user_dict.items():
        interaction = {}
        # import pdb;pdb.set_trace()
        for item_id in item_list:
            interaction[item_id] = item_dict.get(str(item_id), "")  # 若找不到则设为空字符串
        output_dict[user_id] = {
            "interaction sequences": interaction
        }

    # 保存到general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print("已保存到 general_dict.json")





def prompts_generation(prompt_template):
    # 读取 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # 构建 user_prompt_dict
    user_prompt_dict = {}

    for user_id, data in user_data.items():
        item_dict = data["interaction sequences"]
        
        # 构造每个用户的 interaction_sequence 文本块
        # interaction_lines = [f'"{item_id}": "{desc}"' for item_id, desc in item_dict.items()]
        interaction_lines = [f'{item_id}: {desc}' for item_id, desc in item_dict.items()]
        interaction_sequence_str = "\n".join(interaction_lines)

        # 填充模板
        prompt = prompt_template.format(
            user_id=user_id,
            interaction_sequence=interaction_sequence_str
        )

        # 将该用户的 prompt 保存到 user_prompt_dict，直接保存 prompt 字符串
        user_prompt_dict[user_id] = prompt
        # print(f"Generated prompt for user {user_id}: {prompt}")


    # 返回构建的 user_prompt_dict
    return user_prompt_dict

def challenge_prompts_generation(prompt_template, role_A, role_B):
    # 读取 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # 遍历每个用户的数据并更新
    for user_id, data in user_data.items():
        role_B_content = data[role_B+"_Initial_Judgment_Response"]

        # 填充模板
        prompt = prompt_template.format(
            role_A=role_A,
            role_B=role_B,
            role_B_content=role_B_content
        )

        # 更新 user_data 中的相应字段
        user_data[user_id][role_A+"_Challenge_"+role_B+"_Prompt"] = prompt

    # 将更新后的 user_data 写入 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    print("Updated general_dict.json successfully.")

def defend_prompts_generation(role_A, role_B):
    # 读取 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)

    prompt_template = globals()[role_A + "_Defend_Generation_Prompt"]


    # 遍历每个用户的数据并更新
    for user_id, data in user_data.items():
        Content_of_doubt = data[role_B+"_Challenge_"+role_A+"_Response"]

        # 填充模板
        prompt = prompt_template.format(
            role_B=role_B,
            Content_of_doubt=Content_of_doubt
        )

        # 更新 user_data 中的相应字段
        user_data[user_id][role_A+"_Defend_"+role_B+"_Prompt"] = prompt

    # 将更新后的 user_data 写入 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    print("Updated general_dict.json successfully.")


def self_refine_prompts_generation(role):
    # 读取 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # 遍历每个用户的数据并更新
    for user_id, data in user_data.items():
        if role == "Noisy_User_Analyst":
            prompt_template = Noisy_Self_Refine_Prompt
        else:
            prompt_template = Self_Refine_Prompt

        # 更新 user_data 中的相应字段
        user_data[user_id][role+"_Self_Refine_Prompt"] = prompt_template

    # 将更新后的 user_data 写入 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    print("Updated general_dict.json successfully.")

def judge_prompts_generation():
    input_path = "/home/aiding/LoRec/debate/data/general_dict.json"

    # 读取 general_dict.json
    with open(input_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # 遍历每个用户的数据并更新
    for user_id, data in user_data.items():
        item_dict = data.get("interaction sequences", {})

        # 构造 interaction_sequence 字符串
        interaction_lines = [f'{item_id}: {desc}' for item_id, desc in item_dict.items()]
        interaction_sequence_str = "\n".join(interaction_lines)

        clean_content = data.get("Clean_User_Advocate_Self_Refine_Response", "")
        noisy_content = data.get("Noisy_User_Analyst_Self_Refine_Response", "")
        fraduster_content = data.get("Fraud_Investigator_Self_Refine_Response", "")

        prompt = Final_Judge_Prompt.format(
            interaction_sequence=interaction_sequence_str,
            clean_content=clean_content,
            noisy_content=noisy_content,
            fraduster_content=fraduster_content
        )

        # 只添加或更新 Final_Judge_Prompt 字段
        data["Final_Judge_Prompt"] = prompt

    # 写回原文件（保存增量修改）
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    print("Updated Final_Judge_Prompt in general_dict.json successfully.")





def init_llm():
    MODEL_PATH = "/mnt/data/aiding/Qwen2.5-7B-Instruct"
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.8
    )
    return llm



def init_judge_vllm_generation(user_prompt_dict, llm, role, output_path="/home/aiding/LoRec/debate/data/general_dict.json"):
    import logging
    import pytz
    from datetime import datetime
    import json
    from vllm import SamplingParams

    prompt_tag = "{}_Initial_Judgment_Prompt".format(role)
    response_tag = "{}_Initial_Judgment_Response".format(role)

    # 设置日志格式和处理器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])
    
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Starting the inference process at {current_time}...")

    BATCH = 128

    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": -1,
        "max_tokens": 512,
    }
    sampling_params = SamplingParams(**params_dict)

    user_ids = list(user_prompt_dict.keys())
    prompts = list(user_prompt_dict.values())

    results = {}
    batch_input = []
    batch_user_ids = []

    for uid, prompt in zip(user_ids, prompts):
        batch_input.append(prompt)
        batch_user_ids.append(uid)

        if len(batch_input) == BATCH:
            logging.info(f"Processing batch with {BATCH} prompts...")
            start_time = datetime.now(tz)

            outputs = llm.generate(batch_input, sampling_params)
            for user_id, output in zip(batch_user_ids, outputs):
                generated_text = output.outputs[0].text
                results[user_id] = {
                    prompt_tag: user_prompt_dict[user_id],
                    response_tag : generated_text.strip()
                }

            end_time = datetime.now(tz)
            logging.info(f"Batch processed in {end_time - start_time} seconds.")

            logging.info(f"Appending batch results to {output_path}...")
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}

            for user_id, result in results.items():
                if user_id in existing_data:
                    existing_data[user_id].update(result)
                else:
                    existing_data[user_id] = result

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            batch_input = []
            batch_user_ids = []
            results = {}

    if batch_input:
        logging.info(f"Processing last batch with {len(batch_input)} prompts...")
        start_time = datetime.now(tz)

        outputs = llm.generate(batch_input, sampling_params)
        for user_id, output in zip(batch_user_ids, outputs):
            generated_text = output.outputs[0].text
            results[user_id] = {
                prompt_tag: user_prompt_dict[user_id],
                response_tag : generated_text.strip()
            }

        end_time = datetime.now(tz)
        logging.info(f"Last batch processed in {end_time - start_time} seconds.")

        logging.info(f"Appending final results to {output_path}...")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        for user_id, result in results.items():
            if user_id in existing_data:
                existing_data[user_id].update(result)
            else:
                existing_data[user_id] = result

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

    current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Inference process completed at {current_time}. Results saved to {output_path}")



def multi_rounds_prompts_generation(role="Clean_User_Advocate"):    
    # 读取 general_dict.json
    with open("/home/aiding/LoRec/debate/data/general_dict.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # 构建 example_input
    example_input = {}

    # 遍历每个用户的数据
    for user_id, data in user_data.items():
        # 初始化每个用户的对话列表
        conversation = []
        
        # 标志，用来确定轮流添加角色
        user_turn = True  # 假设从 "user" 角色开始

        # 遍历该用户的所有键，筛选出以 role 为前缀的键
        for key, value in data.items():
            if key.startswith(role):
                # 根据轮流的方式来确定角色
                role_type = "user" if user_turn else "assistant"
                
                # 将该消息添加到对话列表
                conversation.append({
                    "role": role_type,  # 根据标志动态设置角色
                    "content": value
                })
                
                # 切换角色
                user_turn = not user_turn

        # 将该用户的对话添加到 example_input 中
        example_input[user_id] = conversation

    return example_input

def multi_rounds_vllm_generation(
    llm,
    role_A: str,
    role_B: str,
    user_prompt_dict: Dict[str, List[dict]],  # 输入格式: {用户ID: 多轮对话消息列表}
    output_path: str = "generated_results.json",
) -> None:
    """vLLM离线批量推理（支持多轮对话）"""
    import os
    import json
    import pytz
    import logging
    from datetime import datetime
    from vllm import SamplingParams

    # 1. 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    tz = pytz.timezone('Asia/Shanghai')
    logging.info(f"Start processing at {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    # 3. 格式化输入
    formatted_prompts = []
    user_ids = []
    for uid, messages in user_prompt_dict.items():
        formatted_prompt = (
            "".join(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                for msg in messages
            ) +
            "<|im_start|>assistant\n"
        )
        formatted_prompts.append(formatted_prompt)
        user_ids.append(uid)

    # 4. 批量推理
    batch_size = 128  # 可调整
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        batch_user_ids = user_ids[i:i + batch_size]

        logging.info(f"Processing batch {i//batch_size + 1} with {len(batch)} prompts...")

        outputs = llm.generate(batch, sampling_params)

        # 构造 batch 结果
        batch_results = {}
        response_tag = role_A+"_Challenge_"+role_B+"_Response"
        for uid, output in zip(batch_user_ids, outputs):
            batch_results[uid] = {
                response_tag: output.outputs[0].text
            }

        # 读取已有文件内容
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # 合并本 batch 的结果
        for user_id, result in batch_results.items():
            if user_id in existing_data:
                existing_data[user_id].update(result)
            else:
                existing_data[user_id] = result

        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Batch {i//batch_size + 1} results saved to {output_path}")


def multi_rounds_defend_vllm_generation(
    llm,
    role_A: str,
    role_B: str,
    user_prompt_dict: Dict[str, List[dict]],  # 输入格式: {用户ID: 多轮对话消息列表}
    output_path: str = "generated_results.json",
) -> None:
    """vLLM离线批量推理（支持多轮对话）"""
    import os
    import json
    import pytz
    import logging
    from datetime import datetime
    from vllm import SamplingParams

    # 1. 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    tz = pytz.timezone('Asia/Shanghai')
    logging.info(f"Start processing at {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    # 3. 格式化输入
    formatted_prompts = []
    user_ids = []
    for uid, messages in user_prompt_dict.items():
        formatted_prompt = (
            "".join(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                for msg in messages
            ) +
            "<|im_start|>assistant\n"
        )
        formatted_prompts.append(formatted_prompt)
        user_ids.append(uid)

    # 4. 批量推理
    batch_size = 128  # 可调整
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        batch_user_ids = user_ids[i:i + batch_size]

        logging.info(f"Processing batch {i//batch_size + 1} with {len(batch)} prompts...")

        outputs = llm.generate(batch, sampling_params)

        # 构造 batch 结果
        batch_results = {}
        response_tag = role_A+"_Defend_"+role_B+"_Response"
        for uid, output in zip(batch_user_ids, outputs):
            batch_results[uid] = {
                response_tag: output.outputs[0].text
            }

        # 读取已有文件内容
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # 合并本 batch 的结果
        for user_id, result in batch_results.items():
            if user_id in existing_data:
                existing_data[user_id].update(result)
            else:
                existing_data[user_id] = result

        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Batch {i//batch_size + 1} results saved to {output_path}")


def multi_rounds_self_refine_vllm_generation(
    llm,
    role: str,
    user_prompt_dict: Dict[str, List[dict]],  # 输入格式: {用户ID: 多轮对话消息列表}
    output_path: str = "generated_results.json",
) -> None:
    """vLLM离线批量推理（支持多轮对话）"""
    import os
    import json
    import pytz
    import logging
    from datetime import datetime
    from vllm import SamplingParams

    # 1. 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    tz = pytz.timezone('Asia/Shanghai')
    logging.info(f"Start processing at {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    # 3. 格式化输入
    formatted_prompts = []
    user_ids = []
    for uid, messages in user_prompt_dict.items():
        formatted_prompt = (
            "".join(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                for msg in messages
            ) +
            "<|im_start|>assistant\n"
        )
        formatted_prompts.append(formatted_prompt)
        user_ids.append(uid)

    # 4. 批量推理
    batch_size = 128  # 可调整
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        batch_user_ids = user_ids[i:i + batch_size]

        logging.info(f"Processing batch {i//batch_size + 1} with {len(batch)} prompts...")

        outputs = llm.generate(batch, sampling_params)

        # 构造 batch 结果
        batch_results = {}
        response_tag = role+"_Self_Refine_Response"
        for uid, output in zip(batch_user_ids, outputs):
            batch_results[uid] = {
                response_tag: output.outputs[0].text
            }

        # 读取已有文件内容
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # 合并本 batch 的结果
        for user_id, result in batch_results.items():
            if user_id in existing_data:
                existing_data[user_id].update(result)
            else:
                existing_data[user_id] = result

        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Batch {i//batch_size + 1} results saved to {output_path}")

def final_judge_vllm_generation(llm, output_path="/home/aiding/LoRec/debate/data/general_dict.json"):
    import logging
    import pytz
    from datetime import datetime
    import json
    from vllm import SamplingParams

    prompt_key = "Final_Judge_Prompt"
    response_key = "Final_Judge_Response"

    # 设置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])
    
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Starting Final Judge Inference at {current_time}...")

    BATCH = 128

    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": -1,
        "max_tokens": 512,
    }
    sampling_params = SamplingParams(**params_dict)

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("general_dict.json not found or invalid.")
        return

    # 构造 batch 输入
    user_ids = []
    prompts = []
    for uid, data in user_data.items():
        if prompt_key in data and response_key not in data:  # 仅处理还没生成过的
            user_ids.append(uid)
            prompts.append(data[prompt_key])

    if not prompts:
        logging.info("No new prompts to process. Exiting.")
        return

    results = {}
    batch_input = []
    batch_user_ids = []

    for uid, prompt in zip(user_ids, prompts):
        batch_input.append(prompt)
        batch_user_ids.append(uid)

        if len(batch_input) == BATCH:
            logging.info(f"Processing batch with {BATCH} prompts...")
            start_time = datetime.now(tz)

            outputs = llm.generate(batch_input, sampling_params)
            for user_id, output in zip(batch_user_ids, outputs):
                generated_text = output.outputs[0].text.strip()
                results[user_id] = generated_text

            end_time = datetime.now(tz)
            logging.info(f"Batch processed in {end_time - start_time} seconds.")

            # 更新 JSON 文件
            for user_id, generated_text in results.items():
                user_data[user_id][response_key] = generated_text

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, indent=2, ensure_ascii=False)

            batch_input = []
            batch_user_ids = []
            results = {}

    if batch_input:
        logging.info(f"Processing last batch with {len(batch_input)} prompts...")
        start_time = datetime.now(tz)

        outputs = llm.generate(batch_input, sampling_params)
        for user_id, output in zip(batch_user_ids, outputs):
            generated_text = output.outputs[0].text.strip()
            results[user_id] = generated_text

        end_time = datetime.now(tz)
        logging.info(f"Last batch processed in {end_time - start_time} seconds.")

        for user_id, generated_text in results.items():
            user_data[user_id][response_key] = generated_text

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)

    current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Final Judge Inference completed at {current_time}. Results saved to {output_path}")





if __name__ == "__main__":
    get_interactions_with_text()
    llm = init_llm()
    # initial judgment
    # 单次手动调节
    # prompts = prompts_generation(prompt_template=Fraud_Investigator_Initial_Judgment_Prompt)    # prompt_template choices = [Clean_User_Advocate_Initial_Judgment_Prompt, Noisy_User_Analyst_Initial_Judgment_Prompt, Fraud_Investigator_Initial_Judgment_Prompt]
    # init_judge_vllm_generation(prompts, llm, role="Fraud_Investigator", output_path="/home/aiding/LoRec/debate/data/general_dict.json")  # role choices= [Noisy_User_Analyst, Clean_User_Advocate, Fraud_Investigator]
    # 三个roles自动完成
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role in roles:
        prompt_template = globals()[f"{role}_Initial_Judgment_Prompt"]
        prompts = prompts_generation(prompt_template=prompt_template)    # prompt_template choices = [Clean_User_Advocate_Initial_Judgment_Prompt, Noisy_User_Analyst_Initial_Judgment_Prompt, Fraud_Investigator_Initial_Judgment_Prompt]
        init_judge_vllm_generation(prompts, llm, role=role, output_path="/home/aiding/LoRec/debate/data/general_dict.json")  # role choices= [Noisy_User_Analyst, Clean_User_Advocate, Fraud_Investigator]

    # challenge generation
    # role_A="Fraud_Investigator"
    # role_B="Noisy_User_Analyst"
    # challenge_prompts_generation(prompt_template=Challenge_Generation_Prompt, role_A=role_A, role_B=role_B)  # role choices= [Noisy_User_Analyst, Clean_User_Advocate, Fraud_Investigator]
    # multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role_A)
    # multi_rounds_vllm_generation(
    #     llm,
    #     role_A=role_A,
    #     role_B=role_B,
    #     user_prompt_dict=multi_rounds_prompt_dict,
    #     output_path="/home/aiding/LoRec/debate/data/general_dict.json"
    # )

    # 三个role自动完成challenge
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role_A in roles:
        for role_B in roles:
            if role_A == role_B:
                continue  # 跳过相同角色组合
            print(f"Running with role_A = {role_A}, role_B = {role_B}")
            challenge_prompts_generation(prompt_template=Challenge_Generation_Prompt, role_A=role_A, role_B=role_B)  # role choices= [Noisy_User_Analyst, Clean_User_Advocate, Fraud_Investigator]
            multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role_A)
            multi_rounds_vllm_generation(
                llm,
                role_A=role_A,
                role_B=role_B,
                user_prompt_dict=multi_rounds_prompt_dict,
                output_path="/home/aiding/LoRec/debate/data/general_dict.json"
            )    


    #defend generation
    # 单次，需指定role
    # role_A="Clean_User_Advocate"
    # role_B="Noisy_User_Analyst"
    # defend_prompts_generation(role_A, role_B)
    # multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role_A)
    # multi_rounds_defend_vllm_generation(
    #     llm,
    #     role_A=role_A,
    #     role_B=role_B,
    #     user_prompt_dict=multi_rounds_prompt_dict,
    #     output_path="/home/aiding/LoRec/debate/data/general_dict.json"
    # )

    # 三个role自动完成defend
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]

    for role_A in roles:
        for role_B in roles:
            if role_A == role_B:
                continue  # 跳过相同角色组合
            print(f"Running with role_A = {role_A}, role_B = {role_B}")
            
            defend_prompts_generation(role_A, role_B)
            
            multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role_A)
            
            output_file = f"/home/aiding/LoRec/debate/data/general_dict.json"
            
            multi_rounds_defend_vllm_generation(
                llm,
                role_A=role_A,
                role_B=role_B,
                user_prompt_dict=multi_rounds_prompt_dict,
                output_path=output_file
            )

    # self-refine generation
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role in roles:
        self_refine_prompts_generation(role)
        multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role)
        multi_rounds_self_refine_vllm_generation(
        llm,
        role,
        multi_rounds_prompt_dict,  # 输入格式: {用户ID: 多轮对话消息列表}
        output_path="/home/aiding/LoRec/debate/data/general_dict.json",
        ) 

    # judge generation
    judge_prompts_generation()
    final_judge_vllm_generation(llm, output_path="/home/aiding/LoRec/debate/data/general_dict.json")


    import pdb;pdb.set_trace()








# MODEL_PATH="/mnt/data/aiding/Qwen2.5-7B-Instruct"
# PROMPT_PATH=""
# BATCH=2


# # 提示列表
# prompts = ["你好啊", "吃饭了没有", "你好，今天天气怎么样？", "孙悟空是谁？"]
# prompt_template = "<|im_start|> user\n{} <|im_end|>"

# # 格式化提示
# prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

# # 采样参数字典
# params_dict = {
#     "n": 1,
#     "best_of": 1,
#     "presence_penalty": 1.0,
#     "frequency_penalty": 0.0,
#     "temperature": 0.5,
#     "top_p": 0.8,
#     "top_k": -1,
#     "max_tokens": 512,
# }

# # 创建一个采样参数对象
# sampling_params = SamplingParams(**params_dict)

# # 创建一个 LLM 模型实例
# # llm = LLM(model=MODEL_PATH, tensor_parallel_size=2, dtype='bfloat16',
# #           trust_remote_code=True, max_model_len=2048, gpu_memory_utilization=0.8)
# llm = LLM(model=MODEL_PATH, tensor_parallel_size=2, dtype='bfloat16',
#           trust_remote_code=True, gpu_memory_utilization=0.8)


# # 从提示生成文本
# batch_input = []
# for prompt in prompts:
#     batch_input.append(prompt)
#     if len(batch_input) == BATCH:
#         outputs = llm.generate(batch_input, sampling_params)
#         # 打印输出结果
#         for output in outputs:
#             prompt = output.prompt
#             print("用户：{}".format(prompt))
#             generated_text = output.outputs[0].text
#             print("AI助手：{}".format(generated_text))
#         batch_input = []
