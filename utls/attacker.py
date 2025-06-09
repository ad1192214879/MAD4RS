import json
import numpy as np

# # random attack

# # === 配置 ===
# # target_items = [1016, 167, 3263, 4004, 2031]
# target_items = [1018, 169, 3265, 4006, 2033]
# # target_items = [1016, 167, 3263, 4004, 2031]
# n_fake_users = 100
# # n_fake_users = 50
# n_inter_per_user = 28  # 总交互数（包括一个目标 item）

# # === 读取 item_dict ===
# with open('/home/aiding/multi-agent-debate-rec/data/Beauty1000dense/item_dict.json', 'r') as f:
#     item_dict = json.load(f)

# all_items = list(map(int, item_dict.keys()))

# # 剔除 target_items，防止重复采样
# candidate_items = list(set(all_items) - set(target_items))

# # === 构造虚假用户 ===
# fake_user_dict = {}

# for uid in range(1, n_fake_users + 1):
#     # 随机选一个目标 item
#     target = np.random.choice(target_items, size=1)[0]
    
#     # 从非目标候选集中采样 27 个，确保与 target 不重复
#     # size = np.random.randint(25, 31)
#     filler_items = np.random.choice(candidate_items, size=n_inter_per_user - 1, replace=False).tolist()
    
#     # 构造交互序列
#     user_items = filler_items + [target]
#     np.random.shuffle(user_items)  # 可选：打乱顺序
    
#     # fake_user_dict[str(uid)] = user_items
#     fake_user_dict[str(uid)] = [int(item) for item in user_items]

# # === 保存到 JSON ===
# with open('/home/aiding/multi-agent-debate-rec/data/Beauty1000dense/fake_user_dict.json', 'w') as f:
#     json.dump(fake_user_dict, f, indent=2)

# print("✅ 生成成功！每个 fake user 包含 1 个目标 item + 27 个随机 item，总共 28 个 item。")


# # attack合并到user_dict
# import json

# # 读取原始 user_dict
# with open('/home/aiding/multi-agent-debate-rec/data/Games1000dense/user_dict.json', 'r') as f:
#     user_dict = json.load(f)

# # 读取 fake_user_dict
# with open('/home/aiding/multi-agent-debate-rec/data/Games1000S3/fake_user_dict.json', 'r') as f:
#     fake_user_dict = json.load(f)

# # 获取新的 user_id 起始编号
# start_id = max(int(uid) for uid in user_dict.keys()) + 1

# # 创建新的 fake_user_dict，user_id 从 start_id 开始重新编号
# new_fake_user_dict = {}
# for i, (old_uid, item_list) in enumerate(fake_user_dict.items()):
#     new_uid = str(start_id + i)
#     new_fake_user_dict[new_uid] = item_list

# # 合并两个字典
# merged_user_dict = {**user_dict, **new_fake_user_dict}

# # 手动格式化输出
# output_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S3/user_dict.json'
# with open(output_path, 'w') as f:
#     f.write('{\n')
#     for idx, (user_id, item_list) in enumerate(sorted(merged_user_dict.items(), key=lambda x: int(x[0]))):
#         f.write(f'\t"{user_id}": [\n')
#         for i, item in enumerate(item_list):
#             comma = ',' if i < len(item_list) - 1 else ''
#             f.write(f'\t\t{json.dumps(item)}{comma}\n')
#         f.write('\t]')
#         if idx < len(merged_user_dict) - 1:
#             f.write(',\n')
#         else:
#             f.write('\n')
#     f.write('}\n')


# # 随机替换
# import json
# import random

# # 设定路径
# item_meta_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S1/item_dict.json'      # 所有合法 item_id 字典
# user_item_path = '/home/aiding/multi-agent-debate-rec/data/Games1000dense/user_dict.json'      # 用户的 item list
# output_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S1/user_dict.json'

# # 读取合法 item_id 字典（假设键是 item_id）
# with open(item_meta_path, 'r') as f:
#     item_meta_dict = json.load(f)

# valid_items = list(item_meta_dict.keys())

# # 将 item_id 变为整型用于比较
# valid_items = [int(i) for i in valid_items if i.isdigit()]  # 防止非数字键
# valid_items_set = set(valid_items)

# # 读取用户的 item list
# with open(user_item_path, 'r') as f:
#     user_item_dict = json.load(f)

# # 执行替换
# for user_id, item_list in user_item_dict.items():
#     item_list_int = [int(i) for i in item_list]  # 转为 int 比较
#     # 判断是否存在 item_id ∈ [1, 400]
#     if any(1 <= iid <= 400 for iid in item_list_int):
#         num_to_replace = max(1, round(0.1 * len(item_list)))  # 至少替换 1 个
#         indices_to_replace = random.sample(range(len(item_list)), num_to_replace)
        
#         for idx in indices_to_replace:
#             old_item = int(item_list[idx])
#             # 选择一个不等于 old_item 的新 item
#             candidate_items = list(valid_items_set - {old_item})
#             new_item = random.choice(candidate_items)
#             item_list[idx] = new_item
#         user_item_dict[user_id] = item_list  # 更新列表（已修改）

# # 保存修改后的字典
# with open(output_path, 'w') as f:
#     json.dump(user_item_dict, f, indent=2)



# # 随机删除
# import json
# import random

# # 设定路径
# user_item_path = '/home/aiding/multi-agent-debate-rec/data/Games1000dense/user_dict.json'  # 用户的 item list
# output_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S2/user_dict.json'        # 输出路径

# # 读取用户的 item list
# with open(user_item_path, 'r') as f:
#     user_item_dict = json.load(f)

# # 执行随机删除
# for user_id, item_list in user_item_dict.items():
#     item_list_int = [int(i) for i in item_list]  # 转为 int 比较
#     # 判断是否存在 item_id ∈ [401, 600]
#     if any(401 <= iid <= 600 for iid in item_list_int):
#         num_to_delete = max(1, round(0.1 * len(item_list)))  # 至少删除 1 个
#         if num_to_delete >= len(item_list):  # 防止删空
#             num_to_delete = len(item_list) - 1
#         indices_to_delete = set(random.sample(range(len(item_list)), num_to_delete))
#         item_list = [item for idx, item in enumerate(item_list) if idx not in indices_to_delete]
#         user_item_dict[user_id] = item_list  # 更新列表（已删除）

# # 保存修改后的字典
# with open(output_path, 'w') as f:
#     json.dump(user_item_dict, f, indent=2)



# import json

# # 设定路径
# user_item_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S1/user_dict.json'
# output_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S1/user_dict_filtered.json'

# # 读取原始 user_dict
# with open(user_item_path, 'r') as f:
#     user_item_dict = json.load(f)

# # 过滤 user_id 在 [1, 400] 范围内的用户（假设 user_id 是字符串格式）
# filtered_user_dict = {
#     uid: items for uid, items in user_item_dict.items()
#     if uid.isdigit() and 1 <= int(uid) <= 400
# }

# # 保存结果
# with open(output_path, 'w') as f:
#     json.dump(filtered_user_dict, f, indent=2)


# import json

# # 设定路径
# user_path1 = '/home/aiding/multi-agent-debate-rec/data/Games1000dense/user_dict.json'
# # user_path1 = '/home/aiding/multi-agent-debate-rec/data/Games1000S1/refer_user_dict.json'
# # user_path1 = '/home/aiding/multi-agent-debate-rec/data/Games1000S1/user_dict_filtered.json'
# user_path2 = '/home/aiding/multi-agent-debate-rec/data/Games1000S3/fake_user_dict.json'
# # user_path2 = '/home/aiding/multi-agent-debate-rec/data/Games1000S2/user_dict_filtered.json'
# # user_path3 = '/home/aiding/multi-agent-debate-rec/data/Games1000S3/fake_user_dict.json'
# output_path = '/home/aiding/multi-agent-debate-rec/data/Games1000S3/refer_user_dict.json'

# # 读取原始 user_dict
# with open(user_path1, 'r') as f1:
#     user_1_dict = json.load(f1)
# with open(user_path2, 'r') as f2:
#     user_2_dict = json.load(f2)
# # with open(user_path3, 'r') as f3:
# #     user_3_dict = json.load(f3)

# for user_id, item_list in user_2_dict.items():
#     # if int(user_id) >= 401:
#     # user_1_dict[user_id] = item_list
#     user_1_dict[str(int(user_id) + 1000)] = item_list

# # for user_id, item_list in user_3_dict.items():
# #     # if int(user_id) >= 401:
# #     # user_1_dict[user_id] = item_list
# #     user_1_dict[str(int(user_id) + 1000)] = item_list

# # 保存结果
# with open(output_path, 'w') as f:
#     json.dump(user_1_dict, f, indent=2)



# import json
# import re
# from collections import defaultdict

# # 路径设置
# input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260521_arts100dense_rounds_qwen3_8b_new_debate_dict.json'

# # 读取 json
# with open(input_path, 'r') as f:
#     data = json.load(f)

# # 正则模式提取 Confidence 值
# confidence_pattern = re.compile(r"Confidence:\s*([0-9.]+)")

# # 保存每个用户的 confidence 值
# user_confidences = defaultdict(dict)

# # 保存每个键名下所有用户的 confidence 列表
# key_confidence_values = defaultdict(list)

# for user_id, response_dict in data.items():
#     for key, value in response_dict.items():
#         # 判断键名是否包含 "Defend" 和 "Response"
#         if "Defend" in key and "Response" in key:
#             match = confidence_pattern.search(value)
#             if match:
#                 confidence = float(match.group(1))
#                 user_confidences[user_id][key] = confidence
#                 key_confidence_values[key].append(confidence)

# # 打印每个键的平均 confidence 值
# print("Average confidence per key:")
# for key, values in key_confidence_values.items():
#     avg_conf = sum(values) / len(values)
#     print(f"{key}: {avg_conf:.4f}")

# # 保存 user_confidences 到文件
# output_path = '/home/aiding/multi-agent-debate-rec/debate/data/user_confidences.json'
# with open(output_path, 'w') as f:
#     json.dump(user_confidences, f, indent=2)




# import json
# import re
# from collections import defaultdict

# # 路径设置
# input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260521_arts100dense_rounds_qwen3_8b_new_debate_dict.json'

# # 读取 json
# with open(input_path, 'r') as f:
#     data = json.load(f)

# # 正则模式提取 Confidence 值
# confidence_pattern = re.compile(r"Confidence:\s*([0-9.]+)")

# # 正则匹配 role 和 round
# role_round_pattern = re.compile(r'^(Clean_User_Advocate|Noisy_User_Analyst|Fraud_Investigator)_Defend_.*_Response_Round(\d+)')

# # 保存置信度
# role_round_confidences = defaultdict(list)

# for user_id, response_dict in data.items():
#     for key, value in response_dict.items():
#         if isinstance(value, str):  # ⬅️ 修复这里的类型检查
#             match_conf = confidence_pattern.search(value)
#             match_key = role_round_pattern.search(key)
#             if match_conf and match_key:
#                 role = match_key.group(1)
#                 round_num = f"Round{match_key.group(2)}"
#                 confidence = float(match_conf.group(1))
#                 role_round_confidences[(role, round_num)].append(confidence)

# # 输出平均值
# print("Average confidence per (role, round):")
# for (role, round_num), values in sorted(role_round_confidences.items()):
#     avg_conf = sum(values) / len(values)
#     print(f"{role} - {round_num}: {avg_conf:.4f}")




# import json
# import re
# from collections import defaultdict, Counter

# # 路径设置
# input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260521_arts100dense_rounds_qwen3_8b_new_debate_dict.json'

# # 读取 json
# with open(input_path, 'r') as f:
#     data = json.load(f)

# # 正则模式提取 Confidence 值
# confidence_pattern = re.compile(r"Confidence:\s*([0-9.]+)")

# # 正则匹配 role 和 round
# role_round_pattern = re.compile(r'^(Clean_User_Advocate|Noisy_User_Analyst|Fraud_Investigator)_Defend_.*_Response_Round(\d+)')

# # 每个 user_id 的每个 round -> role -> confidence
# user_round_role_conf = defaultdict(lambda: defaultdict(dict))

# for user_id, response_dict in data.items():
#     for key, value in response_dict.items():
#         if isinstance(value, str):
#             match_conf = confidence_pattern.search(value)
#             match_key = role_round_pattern.search(key)
#             if match_conf and match_key:
#                 role = match_key.group(1)
#                 round_num = f"Round{match_key.group(2)}"
#                 confidence = float(match_conf.group(1))
#                 user_round_role_conf[user_id][round_num][role] = confidence

# # 统计每个 round 中胜出的 role
# round_winner_counter = Counter()
# total_counts = Counter()

# for user_id, round_dict in user_round_role_conf.items():
#     for round_num, role_conf_dict in round_dict.items():
#         if role_conf_dict:  # 避免空值
#             # 找出最大置信度的角色
#             max_role = max(role_conf_dict.items(), key=lambda x: x[1])[0]
#             round_winner_counter[(round_num, max_role)] += 1
#             total_counts[round_num] += 1

# # 输出每个 round 中，每个 role 获胜的占比
# print("Proportion of each role winning in each round:")
# round_roles = defaultdict(dict)

# for (round_num, role), count in sorted(round_winner_counter.items()):
#     proportion = count / total_counts[round_num]
#     round_roles[round_num][role] = proportion
#     print(f"{round_num} - {role}: {proportion:.4f}")

# # 也可以输出整体统计
# print("\nOverall role win counts:")
# overall_role_count = Counter()
# for (_, role), count in round_winner_counter.items():
#     overall_role_count[role] += count

# total_rounds = sum(overall_role_count.values())
# for role, count in overall_role_count.items():
#     print(f"{role}: {count} ({count / total_rounds:.2%})")

# print("=== Role Win Proportions per Round ===\n")
# rounds = sorted(set(r for r, _ in round_winner_counter.keys()))
# roles = ["Clean_User_Advocate", "Noisy_User_Analyst", "Fraud_Investigator"]

# # 打印表头
# header = "Round".ljust(10) + "".join(role.ljust(25) for role in roles)
# print(header)
# print("-" * len(header))

# # 打印每轮各角色占比
# for rnd in rounds:
#     line = rnd.ljust(10)
#     total = total_counts[rnd]
#     for role in roles:
#         count = round_winner_counter.get((rnd, role), 0)
#         proportion = count / total if total else 0
#         line += f"{proportion:.2%}".ljust(25)
#     print(line)




# import json
# import re
# from collections import defaultdict

# # 路径设置
# input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260519_arts1000dense_rounds_qwen3_8b_new_debate_dict.json'
# # input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260518_arts1000dense_wo_clean_qwen25_32b_debate_dict.json'

# # 正则匹配 Noisy Items
# noisy_items_pattern = re.compile(r"Noisy Items:\s*([0-9,\s]+)")

# # 读取 JSON
# with open(input_path, 'r') as f:
#     data = json.load(f)

# # 保存每个 user 的 round -> noisy_items 集合
# user_noisy_items_per_round = defaultdict(dict)

# # 提取 noisy items
# for user_id, response_dict in data.items():
#     for key, value in response_dict.items():
#         if isinstance(value, str) and "Noisy_User_Analyst" in key and "Response" in key:
#             match = noisy_items_pattern.search(value)
#             if match:
#                 noisy_str = match.group(1)
#                 noisy_items = set(int(x.strip()) for x in noisy_str.strip().split(",") if x.strip())
                
#                 # 提取 round 编号（包括 Initial 特例）
#                 if "Initial_Judgment_Response" in key:
#                     round_id = "Initial"
#                 else:
#                     round_match = re.search(r"Round(\d+)", key)
#                     round_id = f"Round{round_match.group(1)}" if round_match else "Unknown"
                
#                 user_noisy_items_per_round[user_id][round_id] = noisy_items

# # 查找 noisy items 前后有变化的 user_id
# changed_users = []

# for user_id, round_items in user_noisy_items_per_round.items():
#     all_sets = list(round_items.values())
#     if len(all_sets) > 1:
#         first = all_sets[0]
#         for s in all_sets[1:]:
#             if s != first:
#                 changed_users.append(user_id)
#                 break

# # 输出结果
# print("Users with different Noisy Items across rounds:")
# for uid in changed_users:
#     print(uid)

# # （可选）保存变化的用户和他们的 noisy_items 记录
# output_debug_path = '/home/aiding/multi-agent-debate-rec/debate/data/noisy_items_changed_users.json'
# with open(output_debug_path, 'w') as f:
#     json.dump({uid: {rid: list(items) for rid, items in rounds.items()}
#                for uid, rounds in user_noisy_items_per_round.items() if uid in changed_users},
#               f, indent=2)





# import json
# import re
# from collections import defaultdict, Counter

# # 读取 json 数据
# input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260521_arts100dense_rounds_qwen3_8b_new_debate_dict.json'
# with open(input_path, 'r') as f:
#     data = json.load(f)

# # 正则提取
# confidence_pattern = re.compile(r"Confidence:\s*([0-9.]+)")
# full_pattern = re.compile(r'^(?P<defender>Clean_User_Advocate|Noisy_User_Analyst|Fraud_Investigator)_Defend_(?P<target>Clean_User_Advocate|Noisy_User_Analyst|Fraud_Investigator)_Response_Round(?P<round>\d+)')

# # 结构：defender -> target -> round -> confidence
# confidence_by_pair = defaultdict(lambda: defaultdict(dict))

# # 填充结构
# for user_id, response_dict in data.items():
#     for key, value in response_dict.items():
#         if isinstance(value, str):
#             match_conf = confidence_pattern.search(value)
#             match_full = full_pattern.match(key)
#             if match_conf and match_full:
#                 defender = match_full.group("defender")
#                 target = match_full.group("target")
#                 round_num = int(match_full.group("round"))
#                 confidence = float(match_conf.group(1))
#                 confidence_by_pair[defender][target][round_num] = confidence

# # 计算 delta 并统计平均变化
# delta_by_pair = defaultdict(list)

# for defender in confidence_by_pair:
#     for target in confidence_by_pair[defender]:
#         round_conf = confidence_by_pair[defender][target]
#         rounds = sorted(round_conf.keys())
#         prev_conf = 1.0  # 初始 confidence 为 1.0
#         for r in rounds:
#             current_conf = round_conf[r]
#             delta = current_conf - prev_conf
#             delta_by_pair[(defender, target)].append((r, delta))
#             prev_conf = current_conf

# # 输出每个 defender -> target 的每轮变化以及平均变化
# for (defender, target), deltas in delta_by_pair.items():
#     print(f"\n{defender} -> {target}:")
#     for round_num, delta in deltas:
#         print(f"  Round {round_num}: Δ = {delta:.4f}")
#     avg_delta = sum(d for _, d in deltas) / len(deltas)
#     print(f"  Average Δ: {avg_delta:.4f}")




# import json
# import re
# from collections import defaultdict

# # 读取 JSON 数据
# # input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260521_arts100dense_rounds_qwen3_8b_new_debate_dict.json'
# input_path = '/home/aiding/multi-agent-debate-rec/debate/data/20260519_arts1000dense_rounds_qwen3_8b_new_debate_dict.json'


# with open(input_path, 'r') as f:
#     data = json.load(f)

# # 正则表达式
# # confidence_pattern = re.compile(r"Confidence:\s*([0-9.]+)")
# confidence_pattern = re.compile(r"\*?[*]*Confidence:\*?[*]*\s*([0-9.]+)")
# full_pattern = re.compile(r'^(?P<defender>Clean_User_Advocate|Noisy_User_Analyst|Fraud_Investigator)_Defend_(?P<target>Clean_User_Advocate|Noisy_User_Analyst|Fraud_Investigator)_Response_Round(?P<round>\d+)')


# # 结构：user_id -> defender -> target -> round -> confidence
# confidence_by_user = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# # 填充结构
# for user_id, response_dict in data.items():
#     for key, value in response_dict.items():
#         if isinstance(value, str):
#             match_conf = confidence_pattern.search(value)
#             match_full = full_pattern.match(key)
#             if match_conf and match_full:
#                 defender = match_full.group("defender")
#                 target = match_full.group("target")
#                 round_num = int(match_full.group("round"))
#                 confidence = float(match_conf.group(1))
#                 confidence_by_user[user_id][defender][target][round_num] = confidence

# # 计算 delta 并组织为 JSON 结构
# output_data = defaultdict(dict)

# for user_id in confidence_by_user:
#     for defender in confidence_by_user[user_id]:
#         for target in confidence_by_user[user_id][defender]:
#             round_conf = confidence_by_user[user_id][defender][target]
#             rounds = sorted(round_conf.keys())
#             prev_conf = 1.0  # 初始 confidence 为 1.0
#             deltas = []
#             for r in rounds:
#                 current_conf = round_conf[r]
#                 delta = current_conf - prev_conf
#                 deltas.append({"round": r, "delta": round(delta, 4)})
#                 prev_conf = current_conf
#             avg_delta = sum(d["delta"] for d in deltas) / len(deltas) if deltas else 0.0
#             output_data[user_id][f"{defender}_to_{target}"] = {
#                 "deltas": deltas,
#                 "average_delta": round(avg_delta, 4)
#             }

# # 保存到 JSON 文件
# output_path = '/home/aiding/multi-agent-debate-rec/debate/data/confidence_changes.json'
# with open(output_path, 'w') as f:
#     json.dump(output_data, f, indent=2)

# # 计算并打印所有用户的平均 delta
# all_avg_deltas = []
# for user_id in output_data:
#     for pair, info in output_data[user_id].items():
#         all_avg_deltas.append(info["average_delta"])

# overall_avg_delta = sum(all_avg_deltas) / len(all_avg_deltas) if all_avg_deltas else 0.0
# print(f"\nOverall Average Δ Across All Users: {overall_avg_delta:.4f}")



import json
from collections import defaultdict

# 加载 JSON 文件
with open("/home/aiding/multi-agent-debate-rec/debate/data/confidence_changes.json", "r") as f:
    confidence_data = json.load(f)

# 结构：pair -> round -> list of abs(delta)
pair_round_deltas = defaultdict(lambda: defaultdict(list))

# 遍历数据
for user_id, user_data in confidence_data.items():
    for pair_key, pair_info in user_data.items():
        for delta_entry in pair_info["deltas"]:
            round_num = delta_entry["round"]
            delta_val = abs(delta_entry["delta"])
            pair_round_deltas[pair_key][round_num].append(delta_val)

# 计算每个 pair 每轮的平均 delta 绝对值
pair_round_avg_abs = {
    pair: {
        round_num: round(sum(deltas) / len(deltas), 4)
        for round_num, deltas in rounds.items()
    }
    for pair, rounds in pair_round_deltas.items()
}

# 输出结果
for pair in sorted(pair_round_avg_abs.keys()):
    print(f"{pair}:")
    for round_num in sorted(pair_round_avg_abs[pair].keys()):
        print(f"  Round {round_num}: {pair_round_avg_abs[pair][round_num]}")

