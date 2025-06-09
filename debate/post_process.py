

# 将judge_response中的信息提取出来，添加到json文件中

import json
import re

# def extract_info(text):
#     # 正则匹配各字段，允许前后有空格，大小写不敏感
#     # role_match = re.search(r"(?i)Role:\s*(.*?)\.", text)
#     role_match = re.search(r"(?i)Role:\s*(.*?)(?:[\n\.])", text)
#     stance_match = re.search(r"(?i)Stance:\s*(.*?)\n", text)
#     confidence_match = re.search(r"(?i)Confidence:\s*(.*?)\.", text)
#     noisy_items_line = re.search(r"(?i)Noisy Items:\s*([0-9,\s]+)", text)
#     explanation_match = re.search(r"(?i)Explanation:\s*(.*)", text, re.DOTALL)

#     # 如果 noisy_items 行存在，优先使用
#     if noisy_items_line:
#         noisy_items = [int(item.strip()) for item in noisy_items_line.group(1).split(",") if item.strip().isdigit()]
#     else:
#         # 如果没有明确 noisy_items 字段，从正文中找 (IDs xxx)
#         noisy_items_inline = re.findall(r"\(IDs? ([\d,\s]+)\)", text)
#         noisy_items = []
#         for match in noisy_items_inline:
#             noisy_items.extend([int(item.strip()) for item in match.split(",") if item.strip().isdigit()])

#     return {
#         "Role": role_match.group(1).strip() if role_match else None,
#         "Stance": stance_match.group(1).strip() if stance_match else None,
#         "Confidence": confidence_match.group(1).strip() if confidence_match else None,
#         "Noisy Items": noisy_items if noisy_items else None,
#         "Explanation": explanation_match.group(1).strip() if explanation_match else None
#     }

# import re

# def extract_info(text):
#     # 统一预处理：替换方括号为逗号（例如 [352] -> 352）
#     normalized_text = re.sub(r'$$([\d\s,]+)$$', r'\1', text)  # 处理[352]格式
    
#     # 正则匹配各字段（兼容大小写、空格、换行符）
#     role_match = re.search(r"(?i)Role:\s*(.*?)(?:[\n\.])", normalized_text)
#     stance_match = re.search(r"(?i)Stance:\s*(.*?)\n", normalized_text)
#     confidence_match = re.search(r"(?i)Confidence:\s*(.*?)\.", normalized_text)
#     explanation_match = re.search(r"(?i)Explanation:\s*(.*)", normalized_text, re.DOTALL)
    
#     # 改进的Noisy Items提取（兼容三种格式）：
#     noisy_items = []
#     # 情况1：显式声明的"Noisy Items:"行（兼容逗号/空格/方括号）
#     noisy_line_match = re.search(r"(?i)Noisy Items:\s*([$$$$\d\s,]+)", normalized_text)
#     if noisy_line_match:
#         raw_items = re.sub(r'[$$$$\s]', '', noisy_line_match.group(1))  # 移除所有方括号和空格
#         noisy_items.extend([int(x) for x in raw_items.split(",") if x.strip().isdigit()])
    
#     # 情况2：从正文中查找 (IDs xxx) 的隐式声明
#     if not noisy_items:
#         noisy_items_inline = re.findall(r"$IDs? ([\d\s,]+)$", normalized_text)
#         for match in noisy_items_inline:
#             noisy_items.extend([int(x) for x in match.replace(" ", "").split(",") if x.strip().isdigit()])
    
#     # # 情况3：纯数字行（如仅出现"[352]"无前缀）
#     # if not noisy_items:
#     #     standalone_items = re.findall(r'(?<!\d)\b(\d{3,})\b(?!\d)', normalized_text)  # 匹配3位以上独立数字
#     #     noisy_items.extend([int(x) for x in standalone_items if x.isdigit()])

#     return {
#         "Role": role_match.group(1).strip() if role_match else None,
#         "Stance": stance_match.group(1).strip() if stance_match else None,
#         "Confidence": confidence_match.group(1).strip() if confidence_match else None,
#         "Noisy Items": noisy_items if noisy_items else None,
#         "Explanation": explanation_match.group(1).strip() if explanation_match else None
#     }

# import re

# def extract_info(text):
#     # 统一预处理：替换方括号为逗号（例如 [352] -> 352）
#     normalized_text = re.sub(r'\[([\d\s,]+)\]', r'\1', text)  # 注意这里原代码写错了用 `$$`

#     # 放宽匹配规则，兼容 **加粗** 格式
#     role_match = re.search(r"(?i)Role:\s*\**(.*?)\**[\n\.]", normalized_text)
#     stance_match = re.search(r"(?i)Stance:\s*\**(.*?)\**[\n\.]", normalized_text)
#     confidence_match = re.search(r"(?i)Confidence:\s*\**(.*?)\**[\n\.]", normalized_text)
#     explanation_match = re.search(r"(?i)Explanation:\s*(.*)", normalized_text, re.DOTALL)

#     # 改进 Noisy Items 提取
#     noisy_items = []

#     # 情况1：显式 "Noisy Items: **xxx**"
#     noisy_line_match = re.search(r"(?i)Noisy Items:\s*\**([0-9,\s]+)\**", normalized_text)
#     if noisy_line_match:
#         raw_items = noisy_line_match.group(1)
#         noisy_items.extend([int(x.strip()) for x in raw_items.split(",") if x.strip().isdigit()])

#     # 情况2：正文内 IDs xxx
#     if not noisy_items:
#         noisy_items_inline = re.findall(r"IDs?[\s:]*([\d\s,]+)", normalized_text)
#         for match in noisy_items_inline:
#             noisy_items.extend([int(x) for x in match.replace(" ", "").split(",") if x.strip().isdigit()])

#     return {
#         "Role": role_match.group(1).strip() if role_match else None,
#         "Stance": stance_match.group(1).strip() if stance_match else None,
#         "Confidence": confidence_match.group(1).strip() if confidence_match else None,
#         "Noisy Items": noisy_items if noisy_items else None,
#         "Explanation": explanation_match.group(1).strip() if explanation_match else None
#     }

import re

def extract_field(text, label):
    # 优先匹配加粗格式（**内容**）
    match = re.search(rf"(?i){label}:\s*\*\*(.*?)\*\*", text)
    if match:
        return match.group(1).strip()
    # 回退：匹配非加粗格式（以换行或句号结尾）
    match = re.search(rf"(?i){label}:\s*(.*?)(?:[\n\.])", text)
    return match.group(1).strip() if match else None

def extract_info(text):
    # 可选预处理：去除 $ 形式的包裹（如果存在）
    normalized_text = re.sub(r'[$]+([\d\s,]+)[$]+', r'\1', text)  # 处理 $$352$$、$352$ 等

    # 提取字段（兼容加粗格式）
    role = extract_field(normalized_text, "Role")
    stance = extract_field(normalized_text, "Stance")
    confidence = extract_field(normalized_text, "Confidence")

    # 提取 Explanation（可能较长，通常在最后）
    explanation_match = re.search(r"(?i)Explanation:\s*\**(.*?)\**\s*$", normalized_text, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else None

    # 提取 noisy items
    noisy_items = []

    # 显式 Noisy Items 字段（支持 **1840** / [1840] / 1840 形式）
    noisy_line_match = re.search(r"(?i)Noisy Items:\s*\**([$\d\s,\[\]]+)\**", normalized_text)
    if noisy_line_match:
        raw = noisy_line_match.group(1)
        cleaned = re.sub(r'[\[\]\s]', '', raw)  # 去除括号、空格
        noisy_items = [int(x) for x in cleaned.split(',') if x.strip().isdigit()]

    # 隐式形式：如 IDs 1840, 2321
    if not noisy_items:
        inline_matches = re.findall(r"IDs?\s+([\d\s,]+)", normalized_text)
        for match in inline_matches:
            noisy_items += [int(x) for x in match.replace(" ", "").split(",") if x.strip().isdigit()]

    return {
        "Role": role,
        "Stance": stance,
        "Confidence": confidence,
        "Noisy Items": noisy_items if noisy_items else None,
        "Explanation": explanation
    }


# # 替换为你的文件路径
# output_path = input_path = "/home/aiding/multi-agent-debate-rec/debate/data/S1_Games_qwen3_8b_debate_dict.json"
# #  "/home/aiding/LoRec/debate/data/10samples_32B_debate_dict.json"

# # 读取 json 文件
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 处理每个用户
# for user_id, user_info in data.items():
#     text = user_info.get("Final_Judge_Response", "")
#     result = extract_info(text)
#     user_info["final_result"] = result

# # 写回 json 文件
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)




# # 统计roles，confidence，noisy_items的数量分布

# import json
# from collections import Counter

# # 读取数据
# with open("/home/aiding/multi-agent-debate-rec/debate/data/S3_qwen3_8b_debate_dict.json", "r") as f:
#     data = json.load(f)

# # 收集 Role 和 Confidence 字段
# roles = []
# confidences = []
# noisy_lengths = []

# for user_id, user_data in data.items():
#     result = user_data.get("final_result", {})
#     if result:
#         role = result.get("Role")
#         confidence = result.get("Confidence")
#         noisy_items = result.get("Noisy Items", [])
#         if role:
#             roles.append(role)
#         if confidence:
#             confidences.append(confidence)
#         if isinstance(noisy_items, list):
#             noisy_lengths.append(len(noisy_items))

# # 统计频次
# role_counter = Counter(roles)
# confidence_counter = Counter(confidences)

# # 计算总数
# total_roles = sum(role_counter.values())
# total_confidences = sum(confidence_counter.values())

# length_counter = Counter(noisy_lengths)
# total = sum(length_counter.values())

# # 输出占比
# print("\n📌 Role 分布:")
# for role, count in role_counter.items():
#     print(f"{role}: {count} ({count / total_roles:.2%})")

# print("\n📌 Confidence 分布:")
# for conf, count in confidence_counter.items():
#     print(f"{conf}: {count} ({count / total_confidences:.2%})")

# print("\n📌 Noisy Items 长度分布:")
# for length, count in sorted(length_counter.items()):
#     print(f"{length} 个: {count} 用户 ({count / total:.2%})")



# # # 将 noisy_items 提取出来，保存到 json 文件中
# import json
# from collections import Counter

# def save_noisy_items_to_json(input_path, output_path):
#     # 读取数据
#     with open(input_path, "r") as f:
#         data = json.load(f)

#     noisy_items_dict = {}
#     for user_id, user_data in data.items():
#         # if int(user_id) <= 400 or int(user_id) >= 601:
#         # if int(user_id) <= 1000 and int(user_id) >= 601:
#         # if int(user_id) >= 401:
#             result = user_data.get("final_result", {})
#             # import pdb;pdb.set_trace()
#             if result:
#                 role = result.get("Role")
#                 # if role == "Noisy User Analyst" or role == "Noisy_User_Analyst" or role == "Judge":
#                 if role == "Noisy User Analyst" and result.get("Noisy Items", []) is not None:
#                     noisy_items_dict[user_id] = result.get("Noisy Items", [])
#                     # noisy_items_dict[str(int(user_id) + 1000)] = result.get("Noisy Items", [])

#     # 写回 json 文件
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(noisy_items_dict, f, indent=2, ensure_ascii=False)

# input_path = "/home/aiding/multi-agent-debate-rec/check/checkpoints/20260502_arts1000dense_qwen25_32b_debate_dict.json"
# # input_path = "/home/aiding/multi-agent-debate-rec/debate/data/20260519_games1000dense_qwen3_8b_debate_dict.json"
# # input_path = "/home/aiding/multi-agent-debate-rec/debate/data/S1_Games_qwen3_8b_debate_dict.json"
# # input_path = "/home/aiding/multi-agent-debate-rec/debate/data/S2_Games_qwen3_8b_debate_dict.json"
# # output_path = "/home/aiding/multi-agent-debate-rec/data/Games1000S1/attack_noisy_items.json"
# output_path = "/home/aiding/multi-agent-debate-rec/data/Arts1000denoising/noisy_items.json"
# # output_path = "/home/aiding/multi-agent-debate-rec/data/Games1000denoising/noisy_items.json"
# save_noisy_items_to_json(input_path, output_path)



# # 根据noisy_items更新 user_dict
# import json

# # 加载数据
# # with open("/home/aiding/multi-agent-debate-rec/data/Games1000S3/noisy_items.json", "r") as f:
# with open("/home/aiding/multi-agent-debate-rec/data/Games1000S3/noisy_items.json", "r") as f:
# # with open("/home/aiding/multi-agent-debate-rec/data/Games1000S1/noisy_items.json", "r") as f:
#     noisy_items_dict = json.load(f)

# with open("/home/aiding/multi-agent-debate-rec/data/Games1000S3/user_dict.json", "r") as f:
#     user_dict = json.load(f)

# noisy_count = 0
# # 删除对应的 noisy_items，保留长度>=3的
# for user_id, noisy_items in noisy_items_dict.items():
#     if not isinstance(noisy_items, list):
#         continue  # 跳过不是列表的情况

#     # if user_id in user_dict and len(noisy_items) <= 3:
#     if user_id in user_dict:
#         user_items = user_dict[user_id]
#         filtered_items = [item for item in user_items if item not in noisy_items]
        
#         # if len(filtered_items) >= 3:
#         if len(filtered_items) >= 2:
#             user_dict[user_id] = filtered_items
#             noisy_count += len(user_items) - len(filtered_items)
#         # 否则不修改，保留原始 user_items

# # 保存修改后的 user_dict
# with open("/home/aiding/multi-agent-debate-rec/data/Games1000S3/user_dict.json", "w") as f:
#     json.dump(user_dict, f, indent=2)

# print(f"Done. Total noisy items removed: {noisy_count}")




# 删除 user_dict 中 role 是 Fraud Investigator 的用户
import json

# 读取32B推理结果
# with open("/home/aiding/LoRec/debate/data/32B_new_debate_dict.json", "r") as f:
with open("/home/aiding/multi-agent-debate-rec/check/checkpoints/20260430_games1000dense_qwen3_32b_debate_dict.json", "r") as f:
# with open("/home/aiding/multi-agent-debate-rec/debate/data/S3_Games_qwen3_8b_debate_dict.json", "r") as f:
# with open("/home/aiding/multi-agent-debate-rec/debate/data/20260519_games1000dense_qwen3_8b_debate_dict.json", "r") as f:
# with open("/home/aiding/LoRec/debate/data/20260430_arts1000dense_qwen3_32b_debate_dict.json", "r") as f:
    debate_data = json.load(f)

# 读取用户序列数据
# with open("/home/aiding/LoRec/data/arts1000denoising/user_dict.json", "r") as f:
with open("/home/aiding/multi-agent-debate-rec/data/Games1000S3/user_dict.json", "r") as f:
# with open("/home/aiding/LoRec/data/Arts1000dense/user_dict.json", "r") as f:
    user_dict = json.load(f)

# 找出需要删除的 user_id（role 是 Fraud Investigator）
to_delete = set()
for user_id, user_data in debate_data.items():
    result = user_data.get("final_result", {})
    if result.get("Role") == "Fraud Investigator":
        to_delete.add(user_id)

# 删除这些用户
filtered_user_dict = {
    user_id: items
    for user_id, items in user_dict.items()
    if user_id not in to_delete
}

# 重新编号 user_id，从1开始
new_user_dict = {}
for idx, (old_user_id, items) in enumerate(filtered_user_dict.items(), start=1):
    new_user_dict[str(idx)] = items

# 保存结果
with open("/home/aiding/multi-agent-debate-rec/data/Games1000S3/user_dict.json", "w") as f:
# with open("/home/aiding/LoRec/data/arts1000denoising/user_dict.json", "w") as f:
    json.dump(new_user_dict, f, indent=2)

print(f"Done. Final user count: {len(new_user_dict)}")

            



# # 统计noisy_items中target_item的数量
# # target_item = [
# #         38646,
# #         60510,
# #         54053,
# #         2524,
# #         36130
# #     ]
# target_item = [
#         1699,
#         2680,
#         2384,
#         102,
#         1576
#     ]

# # 读取数据
# with open("/home/aiding/LoRec/debate/data/32B_attack_debate_dict.json", "r") as f:
#     data = json.load(f)

# count = 0
# for user_id, user_data in data.items():
#     result = user_data.get("final_result", {})
#     if result:
#         role = result.get("Role")
#         # if role == "Noisy User Analyst" or role == "Noisy_User_Analyst" or role == "Judge":
#         if role == "Noisy User Analyst":
#             noisy_items_list = result.get("Noisy Items", [])
#             if noisy_items_list is not None:
#                 for target in target_item:
#                     if target in noisy_items_list:
#                         print(f"User {user_id} has target item {target} in noisy items.")
#                         count += 1
# print(f"Total users with target items in noisy items: {count}")