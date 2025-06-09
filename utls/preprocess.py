# 采样100个users
import json
import random

def process_json_file(input_path, output_path, sample_size=1000):
    # 1. 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 随机采样（如果数据是字典形式）
    if isinstance(data, dict):
        keys = list(data.keys())
        sampled_keys = random.sample(keys, min(sample_size, len(keys)))
        # sampled_data = {str(i): data[key] for i, key in enumerate(sampled_keys)}
        sampled_data = {str(i+1): data[key] for i, key in enumerate(sampled_keys)}
    # # 2. 取交互序列长度最大的前 sample_size 个用户
    # if isinstance(data, dict):
    #     # 按序列长度排序，取前N个
    #     sorted_items = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
    #     sampled_items = sorted_items[:sample_size]
    #     # 为了统一 key，可以重新编号为 1, 2, ..., N
    #     sampled_data = {str(i+1): v for i, (k, v) in enumerate(sampled_items)}
    
    # 3. 保存处理后的JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)
    
    print(f"成功采样 {len(sampled_data)} 条数据并保存到 {output_path}")

# 使用示例
# input_json = "/home/aiding/LoRec/data/Arts/user_dict.json"    # 原始JSON文件路径
# output_json = "/home/aiding/LoRec/data/arts1000random/user_dict_adv.json"  # 输出文件路径
# input_json = "/home/aiding/LoRec/data/Arts/fake_user_dict.json"    # 原始JSON文件路径
# output_json = "/home/aiding/LoRec/data/arts1000dense/fake_user_dict.json"  # 输出文件路径
# process_json_file(input_json, output_json, sample_size=1000)




import json
import random
from collections import defaultdict
import os

# def process_json_file_advanced(input_path, output_path, sample_users=100, target_interactions=800, margin=10, max_item_repeats=2):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     user_metrics = []
#     for uid, interactions in data.items():
#         if interactions:
#             total = len(interactions)
#             unique = len(set(interactions))
#             ratio = unique / total
#             item_counts = {}
#             for item in interactions:
#                 item_counts[item] = item_counts.get(item, 0) + 1
#             if any(count > max_item_repeats for count in item_counts.values()):
#                 continue
#             user_metrics.append((uid, interactions, ratio))

#     if len(user_metrics) < sample_users:
#         raise ValueError(f"可供采样的用户数量不足（只有 {len(user_metrics)} 个）")

#     user_metrics.sort(key=lambda x: x[2])
#     selected_users = user_metrics[:sample_users]
#     total_interactions = sum(len(interactions) for _, interactions, _ in selected_users)
#     final_data = {}

#     if target_interactions - margin <= total_interactions <= target_interactions + margin:
#         for i, (uid, interactions, _) in enumerate(selected_users):
#             final_data[str(i+1)] = interactions
#         print(f"原始总交互数为 {total_interactions}，在目标范围内，直接保留全部交互数据。")

#     elif total_interactions > target_interactions + margin:
#         allocated = []
#         for _, interactions, _ in selected_users:
#             orig_count = len(interactions)
#             new_count = max(1, int(orig_count * target_interactions / total_interactions))
#             allocated.append(new_count)

#         allocated_sum = sum(allocated)
#         diff = target_interactions - allocated_sum
#         if diff != 0:
#             sorted_idx = sorted(range(len(selected_users)), key=lambda i: selected_users[i][2])
#             idx = 0
#             while diff != 0:
#                 if diff > 0:
#                     allocated[sorted_idx[idx]] += 1
#                     diff -= 1
#                 else:
#                     if allocated[sorted_idx[idx]] > 1:
#                         allocated[sorted_idx[idx]] -= 1
#                         diff += 1
#                 idx = (idx + 1) % len(selected_users)

#         for i, (uid, interactions, _) in enumerate(selected_users):
#             new_count = allocated[i]
#             freq = defaultdict(int)
#             for item in interactions:
#                 freq[item] += 1
#             sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
#             selected_interactions = []
#             for item, count in sorted_items:
#                 remain = new_count - len(selected_interactions)
#                 if remain <= 0:
#                     break
#                 selected_interactions.extend([item] * min(count, remain))
#             if len(selected_interactions) < new_count:
#                 extra = random.sample(interactions, new_count - len(selected_interactions))
#                 selected_interactions.extend(extra)
#             final_data[str(i+1)] = selected_interactions

#         final_total = sum(len(v) for v in final_data.values())
#         print(f"下采样后总交互数为 {final_total}，满足目标 {target_interactions} (允许 ±{margin})")

#     else:
#         for i, (uid, interactions, _) in enumerate(selected_users):
#             final_data[str(i+1)] = interactions
#         print(f"原始总交互数 {total_interactions} 低于目标 {target_interactions}，直接保留全部交互。")

#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_data, f, ensure_ascii=False, indent=4)

#     print(f"成功采样 {len(final_data)} 个用户的数据并保存到 {output_path}")



import json
import random
from collections import defaultdict

def process_json_file_advanced(input_path, output_path, sample_users=100, target_interactions=800, margin=10, max_item_repeats=2):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_metrics = []
    for uid, interactions in data.items():
        if interactions:
            total = len(interactions)
            unique = len(set(interactions))
            ratio = unique / total
            item_counts = {}
            for item in interactions:
                item_counts[item] = item_counts.get(item, 0) + 1
            if any(count > max_item_repeats for count in item_counts.values()):
                continue
            user_metrics.append((uid, interactions, ratio))

    if len(user_metrics) < sample_users:
        raise ValueError(f"可供采样的用户数量不足（只有 {len(user_metrics)} 个）")

    # 先按照 ratio 升序，稠密优先
    user_metrics.sort(key=lambda x: x[2])

    # 初步采样 sample_users 个
    selected_users = user_metrics[:sample_users]
    backup_users = user_metrics[sample_users:]  # 剩余备用用户

    total_interactions = sum(len(interactions) for _, interactions, _ in selected_users)
    print(f"初步采样的总交互数为 {total_interactions}")

    # 如果交互数不足，尝试通过替换补偿
    if total_interactions < target_interactions - margin:
        print(f"总交互数 {total_interactions} 低于目标 {target_interactions}，尝试替换交互数少的用户...")

        # 按交互数升序排列（最少交互的在前）
        selected_users.sort(key=lambda x: len(x[1]))

        replaced = False
        backup_idx = 0

        while total_interactions < target_interactions - margin and backup_idx < len(backup_users):
            # 替换一个当前交互数最少的用户
            removed_user = selected_users[0]
            added_user = backup_users[backup_idx]

            total_interactions = total_interactions - len(removed_user[1]) + len(added_user[1])

            selected_users = selected_users[1:] + [added_user]
            selected_users.sort(key=lambda x: len(x[1]))  # 替换后继续按交互数少排序

            backup_idx += 1
            replaced = True

        if replaced:
            print(f"替换后总交互数为 {total_interactions}")

        if total_interactions < target_interactions - margin:
            print(f"⚠️ 警告：补偿后仍不足目标 ({total_interactions} vs {target_interactions})，请考虑增大 sample_users。")

    final_data = {}

    if target_interactions - margin <= total_interactions <= target_interactions + margin:
        # 在目标范围内，直接保留全部交互
        for i, (uid, interactions, _) in enumerate(selected_users):
            final_data[str(i+1)] = interactions
        print(f"总交互数 {total_interactions} 在目标范围内，直接保留全部交互。")

    elif total_interactions > target_interactions + margin:
        # 需要下采样
        print(f"总交互数 {total_interactions} 超过目标，进行下采样...")

        allocated = []
        for _, interactions, _ in selected_users:
            orig_count = len(interactions)
            new_count = max(1, int(orig_count * target_interactions / total_interactions))
            allocated.append(new_count)

        allocated_sum = sum(allocated)
        diff = target_interactions - allocated_sum

        # 微调分配，使得总和正好为 target_interactions
        if diff != 0:
            sorted_idx = sorted(range(len(selected_users)), key=lambda i: selected_users[i][2])
            idx = 0
            while diff != 0:
                if diff > 0:
                    allocated[sorted_idx[idx]] += 1
                    diff -= 1
                else:
                    if allocated[sorted_idx[idx]] > 1:
                        allocated[sorted_idx[idx]] -= 1
                        diff += 1
                idx = (idx + 1) % len(selected_users)

        for i, (uid, interactions, _) in enumerate(selected_users):
            new_count = allocated[i]
            freq = defaultdict(int)
            for item in interactions:
                freq[item] += 1
            sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            selected_interactions = []
            for item, count in sorted_items:
                remain = new_count - len(selected_interactions)
                if remain <= 0:
                    break
                selected_interactions.extend([item] * min(count, remain))
            if len(selected_interactions) < new_count:
                extra = random.sample(interactions, new_count - len(selected_interactions))
                selected_interactions.extend(extra)
            final_data[str(i+1)] = selected_interactions

        final_total = sum(len(v) for v in final_data.values())
        print(f"下采样后总交互数为 {final_total}，满足目标 {target_interactions} (允许 ±{margin})")

    else:
        # 低于目标且没法补偿，但尽量保留
        for i, (uid, interactions, _) in enumerate(selected_users):
            final_data[str(i+1)] = interactions
        print(f"总交互数 {total_interactions} 低于目标，但保留全部交互。")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"成功采样 {len(final_data)} 个用户的数据并保存到 {output_path}")

def reindex_items(input_json_path, output_json_path=None, mapping_output_path=None):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_items = set()
    for user_items in data.values():
        all_items.update(user_items)

    item_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_items), 1)}
    reindexed_data = {
        user_id: [item_mapping[item] for item in items]
        for user_id, items in data.items()
    }

    output_path = output_json_path or input_json_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reindexed_data, f, indent=4, ensure_ascii=False)

    if mapping_output_path:
        with open(mapping_output_path, 'w', encoding='utf-8') as f:
            json.dump(item_mapping, f, indent=4, ensure_ascii=False)

    print(f"完成重新编码！共处理 {len(item_mapping)} 个item")
    print(f"新JSON已保存到: {os.path.abspath(output_path)}")
    if mapping_output_path:
        print(f"映射字典已保存到: {os.path.abspath(mapping_output_path)}")

    return item_mapping

def update_item_dict_with_mapping(item_dict_path, mapping_path, output_path=None):
    with open(item_dict_path, 'r', encoding='utf-8') as f:
        item_dict = json.load(f)

    with open(mapping_path, 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)

    updated_dict = {
        str(id_mapping[old_id]): item_dict[old_id]
        for old_id in item_dict.keys()
        if old_id in id_mapping
    }

    output_path = output_path or item_dict_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dict, f, indent=4, ensure_ascii=False)

    print(f"成功更新item_dict！共保留 {len(updated_dict)}/{len(item_dict)} 个item")
    print(f"结果已保存到: {output_path}")

# if __name__ == "__main__":
#     # Step 1: 采样并生成 user_dict_adv.json
#     process_json_file_advanced(
#         input_path="/home/aiding/LoRec/data/Beauty/user_dict.json",
#         output_path="/home/aiding/LoRec/data/Beauty1000dense/user_dict_adv.json",
#         # sample_users=100,
#         # target_interactions=840,
#         sample_users=1000,
#         target_interactions=8800,
#         margin=10,
#         max_item_repeats=1
#     )
    
#     # process_json_file(input_path="/home/aiding/LoRec/data/Arts/user_dict.json", output_path="/home/aiding/LoRec/data/arts10/user_dict_adv.json", sample_size=10)

#     # Step 2: 重新编码item ID
#     mapping = reindex_items(
#         input_json_path="/home/aiding/LoRec/data/Beauty1000dense/user_dict_adv.json",
#         output_json_path="/home/aiding/LoRec/data/Beauty1000dense/user_dict.json",
#         mapping_output_path="/home/aiding/LoRec/data/Beauty1000dense/item_mapping.json"
#     )

#     # Step 3: 更新 item_dict 的 key
#     update_item_dict_with_mapping(
#         item_dict_path="/home/aiding/LoRec/data/Beauty/item_dict.json",
#         mapping_path="/home/aiding/LoRec/data/Beauty1000dense/item_mapping.json",
#         output_path="/home/aiding/LoRec/data/Beauty1000dense/item_dict.json"
#     )



# 采样attack users

# import json
# import random
# from collections import defaultdict

# # 读取原始 JSON 文件
# with open("/home/aiding/LoRec/data/Arts/inject_user_random_1.json", "r") as f:
#     data = json.load(f)

# user_data = data["user_data"]            # user_id -> [item_id列表]
# target_items = data["target_item"]       # 包含 5 个 target_item_id

# # 反向索引：target_item_id -> 包含该 item 的 user_id 列表
# target_user_map = defaultdict(list)

# for user_id, item_list in user_data.items():
#     for target_item in target_items:
#         if target_item in item_list:
#             target_user_map[target_item].append(user_id)

# # 每个 target_item 采样 20 个 user_id
# final_users = set()
# for target_item in target_items:
#     candidates = target_user_map[target_item]
#     if len(candidates) < 20:
#         raise ValueError(f"Target item {target_item} only found in {len(candidates)} users, less than 20.")
#     sampled = random.sample(candidates, 20)
#     final_users.update(sampled)

# # 重新编码 user_id（从1开始递增）
# sampled_user_data = {}
# for new_id, original_user_id in enumerate(sorted(final_users), start=1):
#     sampled_user_data[str(new_id)] = user_data[original_user_id]


# # 保存为新 JSON 文件
# with open("/home/aiding/LoRec/data/arts100attack/sampled_users.json", "w") as f:
#     json.dump(sampled_user_data, f, indent=4)



# 从dataset.csv文件中读取数据并构建user_dict
# import csv
# import json

# # 定义读取CSV和构建字典的函数
# def build_user_dict(csv_file):
#     user_dict = {}
    
#     # 打开并读取CSV文件
#     with open(csv_file, mode='r') as file:
#         reader = csv.DictReader(file)
        
#         for row in reader:
#             user_id = str(int(row['user_id']) + 1)
#             item_id = int(row['item_id'])
            
#             # 如果user_id不在字典中，初始化其值为一个空列表
#             if user_id not in user_dict:
#                 user_dict[user_id] = []
            
#             # 将当前item_id加入对应user_id的列表中
#             user_dict[user_id].append(item_id)
    
#     return user_dict

# # 将字典保存为JSON文件
# def save_to_json(user_dict, output_file):
#     with open(output_file, mode='w') as file:
#         json.dump(user_dict, file, ensure_ascii=False, indent=4)

# # 使用函数读取CSV并保存为JSON
# csv_file = '/home/aiding/LoRec/data/Beauty/Beauty.csv'  # 输入文件路径
# output_file = '/home/aiding/LoRec/data/Beauty/user_dict.json'  # 输出文件路径

# user_dict = build_user_dict(csv_file)
# save_to_json(user_dict, output_file)

# print(f"字典已成功保存为 {output_file}")

# # 打乱user_dict.json的顺讯
# import json
# import random

# # 读取user_dict.json文件
# def load_user_dict(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return json.load(file)

# # 打乱user_id顺序
# def shuffle_user_dict(user_dict):
#     # 获取所有的user_id
#     user_ids = list(user_dict.keys())
    
#     # 打乱user_ids的顺序
#     random.shuffle(user_ids)
    
#     # 创建一个新的字典，使用打乱后的user_id顺序
#     shuffled_dict = {user_id: user_dict[user_id] for user_id in user_ids}
    
#     return shuffled_dict

# # 保存打乱后的字典到新的JSON文件
# def save_shuffled_user_dict(shuffled_dict, output_file):
#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(shuffled_dict, file, ensure_ascii=False, indent=4)

# # 文件路径
# user_dict_file = '/home/aiding/LoRec/data/Beauty/user_dict.json'  # 输入user_dict.json文件路径
# shuffled_user_dict_file = '/home/aiding/LoRec/data/Beauty/user_dict.json'  # 输出打乱顺序后的文件路径

# # 读取user_dict并打乱顺序
# user_dict = load_user_dict(user_dict_file)
# shuffled_user_dict = shuffle_user_dict(user_dict)

# # 保存打乱顺序后的字典
# save_shuffled_user_dict(shuffled_user_dict, shuffled_user_dict_file)

# print(f"打乱后的user_dict已保存到 {shuffled_user_dict_file}")




# # 从meta_Beauty.json和Beauty_item2asin.json文件中读取数据并构建item_dict
# import json
# import ast

# # 读取Beauty_item2asin.json文件
# def load_item2asin_mapping(file_path):
#     with open(file_path, 'r') as file:
#         item2asin = json.load(file)
#     return item2asin

# # 读取meta_Beauty.json文件并处理数据
# def process_meta_beauty(meta_file, item2asin):
#     item_dict = {}
    
#     # 打开并遍历meta_Beauty.json
#     with open(meta_file, 'r') as file:
#         for line in file:
#             # import pdb;pdb.set_trace()
#             # meta_data = json.loads(line)  # 每行是一个字典
#             meta_data = ast.literal_eval(line)
            
#             # 获取asin值
#             asin = meta_data.get('asin')
            
#             if asin in item2asin.values():
#                 # 获取title和categories，并组合成字符串
#                 title = meta_data.get('title', '')
#                 categories = meta_data.get('categories', [])
                
#                 # import pdb; pdb.set_trace()  # 调试断点，查看当前状态
                
#                 title_str = f"title: {title}"
                
#                 # 检查categories[0]是否为空列表
#                 if categories and isinstance(categories[0], list) and len(categories[0]) > 0:
#                     category_str = "Categories: " + ';'.join(categories[0])  # 将类别连接成字符串
#                     combined_str = f"{title_str}. {category_str}"
#                 else:
#                     combined_str = title_str  # 如果categories[0]为空列表，直接使用title_str
                
#                 # 获取对应的item_id（Beauty_item2asin.json中的键）
#                 item_id = [k for k, v in item2asin.items() if v == asin][0]
                
#                 # 将新的键值对保存在item_dict中
#                 item_dict[str(int(item_id) + 1)] = combined_str
    
#     return item_dict

# # 保存结果到json文件
# def save_item_dict(item_dict, output_file):
#     with open(output_file, 'w') as file:
#         json.dump(item_dict, file, ensure_ascii=False, indent=4)

# # 文件路径
# meta_file = '/home/aiding/RobustRec/dataset/Beauty/meta_Beauty.json'  # 输入meta_Beauty文件路径
# item2asin_file = '/home/aiding/LoRec/data/Beauty/Beauty_item2asin.json'  # 输入Beauty_item2asin文件路径
# output_file = '/home/aiding/LoRec/data/Beauty/item_dict.json'  # 输出文件路径

# # 加载Beauty_item2asin.json并处理meta_Beauty.json
# item2asin_mapping = load_item2asin_mapping(item2asin_file)
# item_dict = process_meta_beauty(meta_file, item2asin_mapping)

# # 保存结果到item_dict.json
# save_item_dict(item_dict, output_file)

# print(f"处理结果已保存到 {output_file}")







# 通过user_dict统计avg.length和sparsity
import json

# 假设 user_dict 已加载为 Python dict
user_dict = json.load(open('/home/aiding/LoRec/data/Beauty1000dense/user_dict.json'))

def analyze_user_dict(user_dict):
    total_interactions = 0
    item_set = set()

    for item_list in user_dict.values():
        total_interactions += len(item_list)
        item_set.update(item_list)

    num_users = len(user_dict)
    num_items = len(item_set)
    avg_seq_len = total_interactions / num_users if num_users > 0 else 0
    sparsity = 1 - (total_interactions / (num_users * num_items)) if num_users > 0 and num_items > 0 else 0

    return {
        "用户数": num_users,
        "物品数": num_items,
        "总交互数": total_interactions,
        "平均序列长度": avg_seq_len,
        "稀疏度": sparsity
    }

print(analyze_user_dict(user_dict))
