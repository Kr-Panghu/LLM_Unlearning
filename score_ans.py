# import sys
# from transformers import AutoTokenizer, pipeline, set_seed
# import torch
# from peft import AdaLoraConfig, TaskType, get_peft_model
# from torch.optim import AdamW
# from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
# from utils import (
#     compute_kl,
#     create_pku_dataloader_from_dataset,
#     create_truthfulqa_dataloader,
#     get_answer_loss,
#     get_rand_ans_loss,
#     get_truthfulQA_answers_plaintext,
# )

# set_seed(38)

# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# prompt_list = []

# # 打开并读取文件
# with open('question_no_harm.txt', 'r') as file:
#     # 遍历文件的每一行
#     for line in file:
#         # 去除行尾的换行符，并将该行文字添加到列表中
#         prompt_list.append(line.strip())
        
# # prompt = "I don't agree with Black Lives Matter."

# prompt_num = len(prompt_list) # Number of test prompts
# num = 1

# with open('output.txt', 'w') as f:
#     sys.stdout = f

#     baseline_model_name = 'facebook/opt-1.3b'
#     generator = pipeline('text-generation', model=AutoModelForCausalLM.from_pretrained(baseline_model_name).to(device), tokenizer=tokenizer, device=device, do_sample=True, num_return_sequences=num)
#     print(f'Basline LLM: {baseline_model_name}')

#     generated_list = []

#     for k in range(prompt_num):
#         prompt = prompt_list[k]
        
#         text = generator(prompt)
#         # generated_texts = [entry['generated_text'].replace(prompt, '').strip() for entry in text]
#         generated_texts = [entry['generated_text'][len(prompt):] for entry in text]
#         for i in range(num):
#             # print(prompt, repr(generated_texts[i]))
#             tmp = repr(generated_texts[i])
#             generated_list.append(tmp)

#     del generator

#     print(prompt_list)
#     print(generated_list)

#     steps = [200, 500, 1000, 2000]
#     for i in range(3):
#         unlearned_model_name = f'models/opt-1.3b_v{i+1}'
#         print(f'Unlearned LLM: {unlearned_model_name}, learning steps: {steps[i]}')
#         generator_ul = pipeline('text-generation', model=unlearned_model_name, tokenizer=tokenizer, device=device, do_sample=True, num_return_sequences=num)
        
#         generated_list = []
        
#         for k in range(prompt_num):
#             prompt = prompt_list[k]

#             text = generator_ul(prompt)
#             generated_texts = [entry['generated_text'][len(prompt):] for entry in text]
#             for i in range(num):
#                 # print(prompt, repr(generated_texts[i]))
#                 tmp = repr(generated_texts[i])
#                 generated_list.append(tmp)
                
#         print(generated_list)
                
#         del generator_ul
        
# sys.stdout = sys.__stdout__


import sys
from transformers import AutoTokenizer, pipeline, set_seed
import torch
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)

set_seed(38)

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

prompt_list = []

# 打开并读取文件
with open('question_no_harm.txt', 'r') as file:
    # 遍历文件的每一行
    for line in file:
        # 去除行尾的换行符，并将该行文字添加到列表中
        prompt_list.append(line.strip())
        
# prompt = "I don't agree with Black Lives Matter."

prompt_num = len(prompt_list) # Number of test prompts
num = 1

with open('output.txt', 'a') as f:
    sys.stdout = f

    baseline_model_name = 'facebook/opt-1.3b'
    generator = pipeline('text-generation', model=AutoModelForCausalLM.from_pretrained(baseline_model_name).to(device), tokenizer=tokenizer, device=device, do_sample=True, num_return_sequences=num)
    print(f'Basline LLM: {baseline_model_name}')

    steps = [200, 500, 1000, 2000]
    
    unlearned_model_name = f'models/opt-350m_v4'
    print(f'Unlearned LLM: {unlearned_model_name}, learning steps: 2000')
    generator_ul = pipeline('text-generation', model=unlearned_model_name, tokenizer=tokenizer, device=device, do_sample=True, num_return_sequences=num)
    
    generated_list = []
    
    for k in range(prompt_num):
        prompt = prompt_list[k]

        text = generator_ul(prompt)
        generated_texts = [entry['generated_text'][len(prompt):] for entry in text]
        for i in range(num):
            # print(prompt, repr(generated_texts[i]))
            tmp = repr(generated_texts[i])
            generated_list.append(tmp)
            
    print(generated_list)
            
    del generator_ul
        
sys.stdout = sys.__stdout__