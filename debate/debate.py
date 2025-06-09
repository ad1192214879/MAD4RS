import json
# from vllm import LLM, SamplingParams

import json
import logging
from vllm import LLM, SamplingParams
from datetime import datetime
import pytz
from typing import Dict, List


Clean_User_Advocate_Initial_Judgment_Prompt="""Your role: Clean User Advocate. Your objective is to analyze the given user interaction sequence and explain why it appears clean, supporting the claim that the user is a normal, non-malicious user.
Scenario: {dataset} Recommendation Scenario
User ID: {user_id}
Interaction Sequence:
 (The format for each line is <Item ID>: <Item Description>)
{interaction_sequence}
Your Output format:
Stance: The user <User ID> is a normal user, and their interaction sequence is clean.
Confidence: High.
Explanation: <Provide a clear reasoning process, highlighting behavioral consistency, engagement patterns, and other relevant factors that indicate normal user activity.>"""

Noisy_User_Analyst_Initial_Judgment_Prompt="""Your Role: Noisy User Analyst. Your objective is to analyze the given user interaction sequence, identify noisy items, and explain why the user is a normal user despite the presence of noise.
Scenario: {dataset} Recommendation Scenario
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
Scenario: {dataset} recommendation scenario
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

Clean_User_Advocate_Defend_Generation_Prompt="""Now it is the debate stage. Your role is Clean User Advocate. You need to respond to the doubts raised by {role_B} about your views, and adjust your confidence in your views according to the rationality of the doubts. There are three levels of confidence: High, Medium, and Low.
{role_B}'s doubts about you:
{Content_of_doubt}
Output Format:
Stance: The user <User ID> is a normal user, and their interaction sequence is clean.
Confidence: <Adjusted confidence>.
Response: <Response Content>."""

Noisy_User_Analyst_Defend_Generation_Prompt="""Now it is the debate stage. Your role is Noisy User Analyst. You need to respond to the doubts raised by {role_B} about your views, and adjust your confidence in your views according to the rationality of the doubts. There are three levels of confidence: High, Medium, and Low.
{role_B}'s doubts about you:
{Content_of_doubt}
Output Format:
Stance: The user <User ID> is a normal user, but the sequence contains noise items.
Confidence: <Adjusted confidence>.
Noisy Items: <Adjusted list of identified noisy item IDs if needed>.
Response: <Response Content>."""

Fraud_Investigator_Defend_Generation_Prompt="""Now it is the debate stage. Your role is Fraud Investigator. You need to respond to the doubts raised by {role_B} about your views, and adjust your confidence in your views according to the rationality of the doubts. There are three levels of confidence: High, Medium, and Low.
{role_B}'s doubts about you:
{Content_of_doubt}
Output Format:
Stance: The user <User ID> is a fraudster, and the interaction sequence is fabricated by an attacker to include a promotional product.
Confidence: <Adjusted confidence>.
Response: <Response Content>."""


Self_Refine_Prompt="""After several rounds of debate, you can reflect on the final explanation and adjust the confidence. There are three levels of confidence: High, Medium, and Low.
Output Format:
Stance: <Stance>
Confidence: <Adjusted confidence>.
Explanation: <Explanation content>."""

Clean_User_Advocate_Self_Refine_Prompt="""After several rounds of debate, you can reflect on the final explanation and adjust the confidence. There are three levels of confidence: High, Medium, and Low.
Output Format:
Stance: The user <User ID> is a normal user, and their interaction sequence is clean.
Confidence: <Adjusted confidence>.
Explanation: <Explanation content>."""


Noisy_User_Analyst_Self_Refine_Prompt="""After several rounds of debate, you can reflect on the final explanation and adjust the confidence. There are three levels of confidence: High, Medium, and Low.
Output Format:
Stance: The user <User ID> is a normal user, but the sequence contains noise items.
Confidence: <Adjusted confidence>.
Noisy Items: <Adjusted list of identified noisy item IDs if needed>.
Explanation: <Explanation content>."""
# Adjusted list of noisy item IDs, if adjustment is required; otherwise, the original list

Fraud_Investigator_Self_Refine_Prompt="""After several rounds of debate, you can reflect on the final explanation and adjust the confidence. There are three levels of confidence: High, Medium, and Low.
Output Format:
Stance: The user <User ID> is a fraudster, and the interaction sequence is fabricated by an attacker to include a promotional product.
Confidence: <Adjusted confidence>.
Explanation: <Explanation content>."""


Final_Judge_Prompt="""You are a judge tasked with evaluating three distinct assessments of a user's historical interaction sequence. Each role presents a mutually exclusive perspective:
Clean User Advocate believes the user is a normal user, and their interaction sequence is clean.
Noisy User Analyst believes the user is a normal user, but the sequence contains noise items.
Fraud Investigator believes the user is a fraudster, and the interaction sequence is fabricated by an attacker to include a promotional product.

Your role is to determine which role's judgment is most accurate based on the provided inputs. 

User Historical Interaction Sequence:
{interaction_sequence}

Clean User Advocate's View:
{clean_content}

Noisy User Analyst's View:
{noisy_content}

Fraud Investigator's View:
{fraduster_content}

Output your decision in the following format:
Role: <Correct role>.
Stance: <Stance content>.
Noisy Items: <List of identified noisy item IDs, if any>.
Explanation: <Brief explanation for your decision>."""


# # 统一日志配置
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# tz = pytz.timezone('Asia/Shanghai')

import logging
import os
from datetime import datetime
import pytz

# 设置时区
tz = pytz.timezone('Asia/Shanghai')

# 创建日志目录
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# 自动生成日志文件名：如 2025-04-16_15-30-00.log
log_filename = f"{datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(log_dir, log_filename)

# 统一日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),  # 写入日志文件
        logging.StreamHandler()  # 输出到终端
    ]
)

logging.info(f"日志已开始记录，文件路径为：{log_path}")




# 文件路径和模型路径（统一定义）
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/32B_newprompt_attack_debate_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/arts100attack/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/arts100attack/item_dict.json"
# MODEL_PATH = "/mnt/data/aiding/Qwen3-32B"
# MODEL_PATH = "/mnt/data/aiding/Qwen3-30B-A3B"
MODEL_PATH = "/mnt/data/aiding/Qwen2.5-32B-Instruct"
# MODEL_PATH = "/mnt/data/aiding/GLM-4-32B-0414"
# MODEL_PATH = "/mnt/data/aiding/DeepSeek-R1-Distill-Qwen-32B"
# MODEL_PATH = "/mnt/data/aiding/Qwen2.5-72B-Instruct-AWQ"
# MODEL_PATH = "/mnt/data/aiding/Qwen2.5-7B-Instruct"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/100samples_32B_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/32B_no_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/20260429_arts10_qwen3_32b_debate_dict.json"
GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/20260429_arts1000dense_qwen25_32b_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/20260424_mind_qwen32b_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/20260418_dpsk_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/20260417_GLM_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/32B_newprompt_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/32B_attack_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/32B_GLM_attack_debate_dict.json"
# GENERAL_DICT_PATH = "/home/aiding/LoRec/debate/data/32B_GLM_debate_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/game1000dense/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/game1000dense/item_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/mind1000dense/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/mind1000dense/item_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/arts1000dense/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/arts1000dense/item_dict.json"
USER_DICT_PATH = "/home/aiding/LoRec/data/Arts1000dense/user_dict.json"
ITEM_DICT_PATH = "/home/aiding/LoRec/data/Arts1000dense/item_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/Beauty1000dense/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/Beauty1000dense/item_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/arts100test/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/arts100test/item_dict.json"
# USER_DICT_PATH = "/home/aiding/LoRec/data/arts10/user_dict.json"
# ITEM_DICT_PATH = "/home/aiding/LoRec/data/arts10/item_dict.json"
DATASET = "Arts" # Games, Arts, News, Beauty



def init_llm() -> LLM:
    """初始化 LLM 模型"""
    logging.info("Initializing LLM model...")
    llm = LLM(
        model=MODEL_PATH,
        # tensor_parallel_size=2,
        tensor_parallel_size=2,
        dtype='bfloat16',
        # dtype='float16',  # awq使用
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        enforce_eager=True
        # enable_reasoning=False,
        # enable_thinking=False,
        # quantization="awq"  # 量化方式
        # max_model_len=2048,
        # max_num_seqs=2

    )
    logging.info("LLM model initialized successfully.")
    return llm

def get_interactions_with_text():
    """从 user_dict 和 item_dict 构建 general_dict"""
    logging.info("Building general_dict from user_dict and item_dict...")
    with open(USER_DICT_PATH, "r", encoding="utf-8") as f:
        user_dict = json.load(f)

    with open(ITEM_DICT_PATH, "r", encoding="utf-8") as f:
        item_dict = json.load(f)

    output_dict = {}
    for user_id, item_list in user_dict.items():
        interaction = {item_id: item_dict.get(str(item_id), "") for item_id in item_list}
        output_dict[user_id] = {"interaction sequences": interaction}

    with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved to {GENERAL_DICT_PATH}")

def init_judge_prompts_generation(prompt_template: str, role: str) -> Dict[str, str]:
    """生成初始判断的 prompts，并写入 GENERAL_DICT_PATH 中"""
    logging.info("Generating initial judgment prompts...")

    # 读取用户数据
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    user_prompt_dict = {}

    for user_id, data in user_data.items():
        item_dict = data["interaction sequences"]
        interaction_lines = [f'{item_id}: {desc}' for item_id, desc in item_dict.items()]
        interaction_sequence_str = "\n".join(interaction_lines)

        prompt = prompt_template.format(
            dataset = DATASET,
            user_id=user_id,
            interaction_sequence=interaction_sequence_str
        )

        user_prompt_dict[user_id] = prompt
        user_data[user_id][role+"_Initial_Judgment_Prompt"] = prompt  # 写入到原始字典中

    # 将带 prompt 的数据重新写回文件
    with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

    return user_prompt_dict

def challenge_prompts_generation(prompt_template: str, role_A: str, role_B: str):
    """生成挑战阶段的 prompts"""
    logging.info(f"Generating challenge prompts for {role_A} challenging {role_B}...")
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    for user_id, data in user_data.items():
        role_B_content = data.get(f"{role_B}_Initial_Judgment_Response", "")
        prompt = prompt_template.format(
            role_A=role_A.replace("_", " "),
            role_B=role_B.replace("_", " "),
            role_B_content=role_B_content
        )
        user_data[user_id][f"{role_A}_Challenge_{role_B}_Prompt"] = prompt

    with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    logging.info("Updated general_dict.json with challenge prompts.")

def defend_prompts_generation(role_A: str, role_B: str, prompt_template: str):
    """生成防御阶段的 prompts"""
    logging.info(f"Generating defend prompts for {role_A} defending against {role_B}...")
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    for user_id, data in user_data.items():
        content_of_doubt = data.get(f"{role_B}_Challenge_{role_A}_Response", "")
        prompt = prompt_template.format(
            role_B=role_B.replace("_", " "),
            Content_of_doubt=content_of_doubt
        )
        user_data[user_id][f"{role_A}_Defend_{role_B}_Prompt"] = prompt

    with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    logging.info("Updated general_dict.json with defend prompts.")

def self_refine_prompts_generation(role: str):
    """生成自我反思阶段的 prompts"""
    logging.info(f"Generating self-refine prompts for role {role}...")
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    prompt_template = globals().get(f"{role}_Self_Refine_Prompt", Self_Refine_Prompt)
    # if role == "Noisy_User_Analyst":
    #     prompt_template = Noisy_Self_Refine_Prompt
    # else:
    #     prompt_template = Self_Refine_Prompt

    for user_id, data in user_data.items():
        user_data[user_id][f"{role}_Self_Refine_Prompt"] = prompt_template

    with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    logging.info("Updated general_dict.json with self-refine prompts.")

def judge_prompts_generation():
    """生成最终判断的 prompts"""
    logging.info("Generating final judge prompts...")
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    for user_id, data in user_data.items():
        item_dict = data.get("interaction sequences", {})
        interaction_lines = [f'{item_id}: {desc}' for item_id, desc in item_dict.items()]
        interaction_sequence_str = "\n".join(interaction_lines)

        # clean_content = data.get("Clean_User_Advocate_Initial_Judgment_Response", "")
        # noisy_content = data.get("Noisy_User_Analyst_Initial_Judgment_Response", "")
        # fraduster_content = data.get("Fraud_Investigator_Initial_Judgment_Response", "")
        clean_content = data.get("Clean_User_Advocate_Self_Refine_Response", "")
        noisy_content = data.get("Noisy_User_Analyst_Self_Refine_Response", "")
        fraduster_content = data.get("Fraud_Investigator_Self_Refine_Response", "")

        prompt = Final_Judge_Prompt.format(
            interaction_sequence=interaction_sequence_str,
            clean_content=clean_content,
            noisy_content=noisy_content,
            fraduster_content=fraduster_content
        )
        data["Final_Judge_Prompt"] = prompt

    with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)

    logging.info("Updated general_dict.json with final judge prompts.")

def multi_rounds_prompts_generation(role: str) -> Dict[str, List[dict]]:
    """生成多轮对话格式的 prompts"""
    logging.info(f"Generating multi-round prompts for role {role}...")
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    example_input = {}
    for user_id, data in user_data.items():
        conversation = []
        user_turn = True
        for key, value in data.items():
            if key.startswith(role):
                role_type = "user" if user_turn else "assistant"
                conversation.append({"role": role_type, "content": value})
                user_turn = not user_turn
        example_input[user_id] = conversation

    return example_input

def unified_vllm_generation(
    llm: LLM,
    prompts_dict: Dict[str, str | List[dict]],
    response_key: str,
    batch_size: int = 128,
    is_multi_round: bool = False,
    temperature: float = 0,
    top_p: float = 1.0,
    max_tokens: int = 512
):
    """统一的 vLLM 批量推理函数"""
    logging.info(f"Starting unified vLLM generation for response key: {response_key}...")

    # 配置采样参数
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=1.0,
        frequency_penalty=0.0,
        temperature=temperature,
        top_p=top_p,
        top_k=-1,
        max_tokens=max_tokens
    )

    # 格式化输入
    formatted_prompts = []
    user_ids = []
    for uid, prompt_data in prompts_dict.items():
        if is_multi_round:
            # qwen chat template
            formatted_prompt = (
                "".join(
                    f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                    for msg in prompt_data
                ) + "<|im_start|>assistant\n"
            )

            # # glm4 chat template
            # formatted_prompt = (
            #     "".join(
            #         f"<|user|>\n{msg['content']}\n" if msg["role"] == "user"
            #         else f"<|assistant|>\n{msg['content']}\n"
            #         for msg in prompt_data
            #     ) + "<|assistant|>"
            # )

        else:
            formatted_prompt = prompt_data
        formatted_prompts.append(formatted_prompt)
        user_ids.append(uid)

    if not formatted_prompts:
        logging.info("No prompts to process. Exiting.")
        return

    # 加载现有数据
    try:
        with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_data = {}

    # 批量推理
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        batch_user_ids = user_ids[i:i + batch_size]

        logging.info(f"Processing batch {i//batch_size + 1} with {len(batch)} prompts...")
        start_time = datetime.now(tz)

        # outputs = llm.chat(batch, sampling_params, chat_template_kwargs={"enable_thinking": False})
        outputs = llm.generate(batch, sampling_params)
        batch_results = {}
        for uid, output in zip(batch_user_ids, outputs):
            generated_text = output.outputs[0].text.strip()
            batch_results[uid] = generated_text

        end_time = datetime.now(tz)
        logging.info(f"Batch processed in {end_time - start_time} seconds.")

        # 更新 user_data
        for uid, generated_text in batch_results.items():
            if uid in user_data:
                user_data[uid][response_key] = generated_text
            else:
                user_data[uid] = {response_key: generated_text}

        # 写入文件
        with open(GENERAL_DICT_PATH, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Batch {i//batch_size + 1} results saved to {GENERAL_DICT_PATH}")

    logging.info(f"vLLM generation for {response_key} completed.")

if __name__ == "__main__":

    # 记录整个流水线的开始时间
    pipeline_start_time = datetime.now(tz)
    logging.info("CONFIGURATION")
    logging.info(f"MODEL_PATH: {MODEL_PATH}")
    logging.info(f"GENERAL_DICT_PATH: {GENERAL_DICT_PATH}")
    logging.info(f"USER_DICT_PATH: {USER_DICT_PATH}")
    logging.info(f"ITEM_DICT_PATH: {ITEM_DICT_PATH}")
    logging.info("Starting the entire debate pipeline...")

    # 初始化 LLM
    start_time = datetime.now(tz)
    llm = init_llm()
    end_time = datetime.now(tz)
    logging.info(f"Step 0: LLM Initialization took {end_time - start_time} seconds.")

    # 主函数：运行完整的辩论流水线
    # Step 1: 构建 general_dict，从 user_dict 和 item_dict 中获取文本序列数据
    start_time = datetime.now(tz)
    get_interactions_with_text()
    end_time = datetime.now(tz)
    logging.info(f"Step 1: Building general_dict took {end_time - start_time} seconds.")

    # Step 2: 初始判断阶段
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role in roles:
        start_time = datetime.now(tz)
        logging.info(f"Starting Initial Judgment for role: {role}")
        prompt_template = globals()[f"{role}_Initial_Judgment_Prompt"]
        prompts = init_judge_prompts_generation(prompt_template=prompt_template, role=role)
        unified_vllm_generation(
            llm=llm,
            prompts_dict=prompts,
            response_key=f"{role}_Initial_Judgment_Response",
            batch_size=256,
            is_multi_round=False,
            temperature=0,    # 默认为0
            # top_p=0.8,  # 默认为1
            max_tokens=512,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        end_time = datetime.now(tz)
        logging.info(f"Step 2: Initial Judgment for {role} took {end_time - start_time} seconds.")

    # Step 3: 挑战阶段
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role_A in roles:
        for role_B in roles:
            if role_A == role_B:
                continue
            start_time = datetime.now(tz)
            logging.info(f"Running challenge phase: {role_A} vs {role_B}")
            challenge_prompts_generation(
                prompt_template=Challenge_Generation_Prompt,
                role_A=role_A,
                role_B=role_B
            )
            multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role_A)
            unified_vllm_generation(
                llm=llm,
                prompts_dict=multi_rounds_prompt_dict,
                response_key=f"{role_A}_Challenge_{role_B}_Response",
                batch_size=256,
                is_multi_round=True,
                temperature=0,
                # top_p=0.9,
                max_tokens=512
            )
            end_time = datetime.now(tz)
            logging.info(f"Step 3: Challenge phase ({role_A} vs {role_B}) took {end_time - start_time} seconds.")

    # Step 4: 防御阶段
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role_A in roles:
        for role_B in roles:
            if role_A == role_B:
                continue
            start_time = datetime.now(tz)
            logging.info(f"Running defend phase: {role_A} vs {role_B}")
            defend_prompts_generation(
                role_A=role_A,
                role_B=role_B,
                prompt_template=globals()[f"{role_A}_Defend_Generation_Prompt"]
            )
            multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role_A)
            unified_vllm_generation(
                llm=llm,
                prompts_dict=multi_rounds_prompt_dict,
                response_key=f"{role_A}_Defend_{role_B}_Response",
                batch_size=256,
                is_multi_round=True,
                temperature=0,
                # top_p=0.9,
                max_tokens=512
            )
            end_time = datetime.now(tz)
            logging.info(f"Step 4: Defend phase ({role_A} vs {role_B}) took {end_time - start_time} seconds.")

    # Step 5: 自我反思阶段
    roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]
    for role in roles:
        start_time = datetime.now(tz)
        logging.info(f"Running self-refine phase for {role}")
        self_refine_prompts_generation(role=role)
        multi_rounds_prompt_dict = multi_rounds_prompts_generation(role=role)
        unified_vllm_generation(
            llm=llm,
            prompts_dict=multi_rounds_prompt_dict,
            response_key=f"{role}_Self_Refine_Response",
            batch_size=256,
            is_multi_round=True,
            temperature=0,
            # top_p=0.9,
            max_tokens=512
        )
        end_time = datetime.now(tz)
        logging.info(f"Step 5: Self-refine phase for {role} took {end_time - start_time} seconds.")

    # Step 6: 最终判断阶段
    start_time = datetime.now(tz)
    logging.info("Running final judge phase...")
    judge_prompts_generation()
    with open(GENERAL_DICT_PATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)
    prompts_dict = {uid: data["Final_Judge_Prompt"] for uid, data in user_data.items() if "Final_Judge_Prompt" in data}
    unified_vllm_generation(
        llm=llm,
        prompts_dict=prompts_dict,
        response_key="Final_Judge_Response",
        batch_size=256,
        is_multi_round=False,
        temperature=0,
        # top_p=0.8,
        max_tokens=512
    )
    end_time = datetime.now(tz)
    logging.info(f"Step 6: Final Judge phase took {end_time - start_time} seconds.")

    # 记录整个流水线的总耗时
    pipeline_end_time = datetime.now(tz)
    total_duration = pipeline_end_time - pipeline_start_time
    logging.info(f"Total pipeline execution time: {total_duration} seconds.")
    