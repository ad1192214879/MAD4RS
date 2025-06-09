# # offfline batch inference
# from vllm import LLM, SamplingParams

# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

# llm = LLM(model="/mnt/data/aiding/Qwen2.5-7B-Instruct")

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# import pdb;pdb.set_trace()


# # 假多轮对话，每次对话还需要把history放在prompt里面
# from vllm import LLM,SamplingParams

# llm = LLM(model="/mnt/data/aiding/Qwen2.5-7B-Instruct")
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

# def chat_with_model(user_input, history=None, keep_history=True):
#     if history is None:
#         history = []

#     # 如果选择保留历史，将当前输入添加到历史中
#     if keep_history:
#         history.append(user_input)

#     # 准备提示词，包含历史对话（如果有）和当前输入
#     if keep_history and history:
#         prompt = "\n".join(history) + "\nAssistant:"
#     else:
#         prompt = user_input + "\nAssistant:"

#     # 生成回复
#     outputs = llm.generate([prompt], sampling_params)
#     response = outputs[0].outputs[0].text.strip()

#     # 如果选择保留历史，将模型的回复也添加到历史中
#     if keep_history:
#         history.append(f"Assistant: {response}")

#     return response, history


# # 示例：进行多轮对话，保留历史
# history = []
# user_input = "你好，你是谁？"
# response, history = chat_with_model(user_input, history, keep_history=True)
# print("Assistant:", response)

# user_input = "你能做什么？"
# response, history = chat_with_model(user_input, history, keep_history=True)
# print("Assistant:", response)

# # 示例：开始新的对话，不保留历史
# user_input = "今天天气怎么样？"
# response, _ = chat_with_model(user_input, keep_history=False)
# print("Assistant:", response)



# # online openai api
# from openai import OpenAI
# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8080/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# chat_response = client.chat.completions.create(
#     model="/mnt/data/aiding/Qwen2.5-7B-Instruct",
#     # model="/mnt/data/aiding/Qwen2.5-7B-Instruct",
#     messages=[
#         {"role": "user", "content": """Your role: Clean User Advocate. Your objective is to analyze the given user interaction sequence and explain why it appears clean, supporting the claim that the user is a normal, non-malicious user.
# Scenario: art recommendation scenario
# User ID: 49562
# Interaction Sequence:
#  (The format for each line is "<Item ID>": "<Item Description>")
# "8151": "Title: Leather Factory Perma Lok Jumbo Lacing Needle-For 1/8&quot;, 5/32&quot;Categories: Crafting;Leathercraft;Lacing Needles",
# "53649": "Title: Tandy Leather Long Jumbo Perma-Lok Needle 1193-05Categories: Crafting;Leathercraft;Lacing Needles",
# "7817": "Title: Perma Lok Super Jumbo Lacing Needle For 1/8&quot;, 5/32&quot; Or 1/4&quot; Lace -- TWO PackCategories: Crafting;Leathercraft;Leathercraft Accessories",
# "47688": "Title: Moleskine Passion Journal Hard Cover Notebook, Wellness, Large (5&quot; x 8.25&quot;) - Passion Journal for Wellness Journaling, Wellness Book with Tab Organization, Track Healthy Lifestyle GoalsCategories: Crafting;Paper & Paper Crafts",
# "27908": "Title: Loopilops SRSS, PurpleCategories: Knitting & Crochet;Crochet Kits",
# "49149": "Title: Ozera 6 Cavities Silicone Soap Mold (2 Pack), Baking Mold Cake Pan, Biscuit Chocolate Mold, Ice Cube TrayCategories: Crafting;Soap Making;Molds".
# Your Output format:
# Stance: The user <User ID> is a normal user, and their interaction sequence is clean.
# Confidence: Very high.
# Explanation: <Provide a clear reasoning process, highlighting behavioral consistency, engagement patterns, and other relevant factors that indicate normal user activity.>
# """},
#     ],
#     temperature=0.7,
#     top_p=0.8,
#     max_tokens=512,
#     extra_body={
#         "repetition_penalty": 1.05,
#     },
# )
# print("Chat response:", chat_response)








# openai api 多轮对话
# from openai import OpenAI

# # 配置 OpenAI 客户端，指向本地 vLLM API 服务
# client = OpenAI(
#     api_key="EMPTY",  # vLLM 默认不需要 API 密钥
#     base_url="http://localhost:8080/v1"  # vLLM API 服务地址
# )

# # 初始化对话历史
# conversation_history = []

# def chat_with_model(user_input):
#     global conversation_history

#     # 将用户输入添加到对话历史
#     conversation_history.append({"role": "user", "content": user_input})

#     # 调用 vLLM API，传递完整的对话历史
#     response = client.chat.completions.create(
#         model="/mnt/data/aiding/Qwen2.5-7B-Instruct",  # 模型路径
#         messages=conversation_history,  # 传递对话历史
#         temperature=0.7,
#         top_p=0.8,
#         max_tokens=512,
#         extra_body={
#             "repetition_penalty": 1.05,
#         },
#     )

#     # 获取模型的回复
#     model_reply = response.choices[0].message.content

#     # 将模型回复添加到对话历史
#     conversation_history.append({"role": "assistant", "content": model_reply})

#     return model_reply

# # 示例对话
# user_input = "请记住我是谁，我叫华晨宇。"
# print("User:", user_input)
# print("Assistant:", chat_with_model(user_input))

# user_input = "我是谁？"
# print("User:", user_input)
# print("Assistant:", chat_with_model(user_input))



# qwen多轮对话（不加速版）
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 加载模型和分词器
# model_name = "/mnt/data/aiding/Qwen2.5-7B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 初始化对话历史
# messages = []

# def chat_with_qwen(user_input, reset_history=False):
#     global messages
#     if reset_history:
#         messages = []
    
#     # 将用户输入添加到对话历史
#     messages.append({"role": "user", "content": user_input})
    
#     # 将对话历史转换为模型输入格式
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
#     # 生成回复
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512
#     )
#     generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]  # 获取新生成的部分
#     response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
#     # 将模型回复添加到对话历史
#     messages.append({"role": "assistant", "content": response})
    
#     return response

# # 示例对话
# user_input = "请告诉我如何与你进行多轮对话。"
# response = chat_with_qwen(user_input)
# print("Assistant:", response)

# # 用户的下一次输入
# user_input = "你能再详细说明一下吗？"
# response = chat_with_qwen(user_input)
# print("Assistant:", response)

# # 开始新的对话
# user_input = "我们换个话题吧。"
# response = chat_with_qwen(user_input, reset_history=True)
# print("Assistant:", response)


# import pdb;pdb.set_trace()