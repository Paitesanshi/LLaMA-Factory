import csv
import pandas as pd
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from trl import AutoModelForCausalLMWithValueHead
import uvicorn, json, datetime
import torch
import os
from tqdm import tqdm
import numpy as np
from vllm import LLM,SamplingParams
import re
# torch.manual_seed(2024)
import requests
import random

def call(prompt):
    payload = {
            "model": "glm4",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "do_sample": True,
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
    }
    URL="http://localhost:8000/v1/chat/completions"
    header={"Content-Type": "application/json"}
    response = requests.post(URL, headers=header, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"Error occurred: {response.text}")
        return response.text

def extract_scores(response):
    #Pattern to match the scores and reasons
    #regex = r"Rationality: \[?(\d+)\]?\nConsistency: \[?(\d+)\]?\nImmersion: \[?(\d+)\]?\nVividness: \[?(\d+)\]?"
    # regex=r"Knowledge Accuracy: \[?(\d+)\]?.*?Emotional Expression: \[?(\d+)\]?.*?Personality Traits: \[?(\d+)\]?.*?Behavioral Accuracy: \[?(\d+)\]?.*?Immersion: \[?(\d+)\]?.*?Adaptability: \[?(\d+)\]?.*?Behavioral Coherence: \[?(\d+)\]?"
    regex = r"Knowledge Accuracy:\s*\[?\s*(\d+)\s*\]?.*?Emotional Expression:\s*\[?\s*(\d+)\s*\]?.*?Personality Traits:\s*\[?\s*(\d+)\s*\]?.*?Behavioral Accuracy:\s*\[?\s*(\d+)\s*\]?.*?Immersion:\s*\[?\s*(\d+)\s*\]?.*?Adaptability:\s*\[?\s*(\d+)\s*\]?.*?Behavioral Coherence:\s*\[?\s*(\d+)\s*\]?"

    # Search for matches
    match = re.search(regex, response, re.DOTALL)

    # Extract the scores if found
    if match:
        accuracy, expression,traits,behavior,immersion,adaptability,coherence = match.groups()
    else:
        accuracy, expression,traits,behavior,immersion,adaptability,coherence = -1, -1,-1,-1,-1,-1,-1

    return [int(accuracy),int(expression), int(traits),int(behavior),int(immersion),int(adaptability),int(coherence)]


# model_dir="/home/v-leiwang8/ChatGLM3/finetune_demo/output/lora_voldemort/checkpoint-8000"
# model_dir="/home/v-leiwang8/sft/LLaMA-Factory/saves/chatglm_rm_text/lora/sft_critic"
#model_dir="/home/v-leiwang8/sft_july/LLaMA-Factory/saves/glm4/lora/sft_2"
#model_dir="THUDM/glm-4-9b-chat"
# model_dir="/home/v-leiwang8/sft_july/LLaMA-Factory/saves/glm4/lora/sft_2"
model_dir="/home/v-leiwang8/sft_july/LLaMA-Factory/saves/glm4/lora/sft_new"
print("Loading model from",model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
# max_model_len, tp_size = 131072, 1
model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True).half().cuda()
model.eval()
# llm = LLM(
#     model=model_dir,
#     tensor_parallel_size=tp_size,
#     max_model_len=max_model_len,
#     trust_remote_code=True,
#     enforce_eager=True,
#     # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
#     # enable_chunked_prefill=True,
#     # max_num_batched_tokens=8192
# )
#file_path="/home/v-leiwang8/sft_july/LLaMA-Factory/data/reward/evaluation_critic_human.json"
#file_path="/home/v-leiwang8/sft/LLaMA-Factory/data/reward/evaluation_critic.json"
file_path="/home/v-leiwang8/sft_july/LLaMA-Factory/data/reward/evaluation_critic_short_new.json"
with open(file_path,'r',encoding='utf-8') as f:
    data=json.load(f)
# data=data[:100]
# data=random.sample(data,500)
file_name="glm4_new_all_710"
csv_path=f"{file_name}_reward_result.csv"
fieldnames = ['Title','Judger','Narrator','Model', 'Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence','Average']
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

#record=pd.DataFrame(columns=fieldnames)

zh_titles=['西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖']
en_titles=['Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]
# zh_titles=['西游记']
# en_titles=['Harry_Potter']
records=[]
#zh_record
#en_record={}
preds=[]
labels=[]
cnt=0
batch_size=1
critic="""
<Note>:
Please identify any issues based on these aspects:
1. Factual Accuracy: Identify and point out any elements that do not accurately match the historical or factual backdrop.
2. Character Consistency: Explicitly highlight inconsistencies between the character's actions, dialogues, and their predefined traits and goals.
3. Logical Coherence: Point out any logical fallacies or actions that contradict the established context or character logic.
4. Content Redundancy: Identify repetitions in dialogue or action that could detract from engagement and realism.
5. Emotional Expression: Assess whether emotional responses and expressions are appropriate and convincingly portrayed, highlighting any discrepancies.
6. Interaction Adaptability: Critique the character's interactions with others, noting any unnatural or contextually inappropriate responses.
7. Creativity and Originality: Evaluate the creativity of responses and actions, pointing out generic or unoriginal content.
8. Detail Handling: Scrutinize the level of detail in scene setting and character enactment, marking areas lacking depth or accuracy.
9. Style Consistency: Ensure that the narrative and linguistic style remains consistent, identifying any deviations.
10. Fluency and Quality: Critically assess the smoothness and quality of the text, highlighting any grammatical errors or awkward phrasings.

"""
response_format=""""
[Response Format]:
Knowledge Accuracy: [1-5]
Emotional Expression: [1-5]
Personality Traits: [1-5]
Behavioral Accuracy: [1-5]
Immersion: [1-5]
Adaptability: [1-5]
Behavioral Coherence: [1-5]

[Response Format Example]:
Knowledge Accuracy: 3
Emotional Expression: 3
Personality Traits: 3
Behavioral Accuracy: 3
Immersion: 3
Adaptability: 3
Behavioral Coherence: 3

[Response]:
"""

new_data=[]
if os.path.exists(f"{file_name}_datas.json"):
    with open("datas.json",'w',encoding='utf-8') as f:
        try:
            lines=f.readlines()
        except:
            lines=[]
        for line in lines:
            data=json.loads(line)
            new_data.append(data)
else:
    with open(f"{file_name}_datas.json",'w',encoding='utf-8') as f:
        print("Create new file")

title=""
for i in tqdm(range(len(new_data), len(data),batch_size)):
    if title!=data[i]['title']:
        print("Processing",data[i]['title'])
        title=data[i]['title']
    batch_prompts = [data[j]['instruction'] for j in range(i, min(i+batch_size, len(data)))]
    #batch_prompts=[]
    # for j in range(i,min(i+batch_size,len(data))):
    #     original_string=data[j]['instruction']
    #     index = original_string.find("<Criteria>:")

    #     # 在 "<Score>:" 之前插入新的字符串
    #     modified_string = original_string[:index] + critic + original_string[index:]

    #     #prompt=data[j]['instruction']+critic
    #     batch_prompts.append(modified_string)
    #batch_prompts=[data[j]['instruction'] for j in range(i,min(i+batch_size,len(data)))]
    #print("batch_prompts:")
    #print(batch_prompts)
    # model_inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt", max_length=8192).to("cuda")

    responses = []
    with torch.no_grad():
        
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": data[i]['instruction']}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

        inputs = inputs.to(model.device)
        gen_kwargs = {"max_length": 10000, "do_sample": True, "top_k": 1}
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response=tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
        data[i]['pred']=response
        data[i]['id']=i
        #new_data.append(data[i])
        with open(f"{file_name}_datas.json",'a',encoding='utf-8') as f:
            json.dump(data[i],f,ensure_ascii=False)
            f.write("\n")
    # stop_token_ids = [151329, 151336, 151338]
    # sampling_params = SamplingParams(temperature=0.95, max_tokens=8192, stop_token_ids=stop_token_ids)

    # inputs = tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True)
    # outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

    # for output in outputs:
    #     #print(output.outputs)
    #     responses.append(output.outputs[0].text)
        
    for j in range(len(responses)):
        pos=responses[j].find(response_format)
        responses[j]=responses[j][pos+len(response_format):]
    for idx, response in enumerate(responses):
        #print(response)
        # print("*"*50)
        # print("response:", response)
        # print("*"*50)
        pred = extract_scores(response)
        label = extract_scores(data[i+idx]['output'])
        if -1 in pred:
            print("Response:",response)
            print("Error: Could not extract scores from response", i+idx)
            continue
        preds.append(pred)
        labels.append(label)
        # print("pred:", pred)
        # print("label:", label)
        # input()
        # Construct a record for each response
        temp = {'id':i+idx,'Title': data[i+idx]['title'], 'Judger': data[i+idx]['judger'], 'Narrator': data[i+idx]['narrator'], 'Model': data[i+idx]['model']}
        criteria = ['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
        for j, criterion in enumerate(criteria):
            temp[criterion] = pred[j]
        temp['Average'] = sum(pred) / len(pred)
        records.append(temp)
    print("Processed", i+batch_size, "records")
    temp_preds=np.array(preds)
    temp_labels=np.array(labels)
    rmse=np.sqrt(np.mean((temp_preds-temp_labels)**2))
    print(rmse)
# for i in tqdm(range(len(data))):
#     prompt=data[i]['instruction']
#     input_ids = tokenizer.encode(text=prompt, add_special_tokens=False) + [tokenizer.eos_token_id]
#     if len(input_ids) > 8192:
#         input_ids = input_ids[-8192:]
#     input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
#     with torch.no_grad():
#         model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
#         generated_ids = model.generate(**model_inputs,pad_token_id=tokenizer.eos_token_id,max_length=8192)

#         response=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         print(response)
#         pred= extract_scores(response)
#         label=extract_scores(data[i]['output'])
#         print("pred:",pred)
#         print("label:",label)
#         preds.extend(pred)
#         lables.extend(label)
        

#     cnt+=1
#     criterion=data[i]['criteria']
#     temp={'Title':data[i]['title'],'Judger':data[i]['judger'],'Narrator':data[i]['narrator'],'Model':data[i]['model'],'Knowledge Accuracy':pred[0],'Emotional Expression':pred[1],'Personality Traits':pred[2],'Behavioral Accuracy':pred[3],'Immersion':pred[4],'Adaptability':pred[5],'Behavioral Coherence':pred[6],'Average':sum(pred)/7}
#     records.append(temp)

with open(f"{file_name}_result_records.json",'w',encoding='utf-8') as f:
    json.dump(records,f,ensure_ascii=False,indent=4)

with open(f"{file_name}_data_with_pred.json",'w',encoding='utf-8') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)
    
df = pd.DataFrame(records)

# 为每条记录计算评估指标的平均值
metrics = ['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
df[metrics] = df[metrics]
df['Average'] = df[metrics].mean(axis=1)

# 创建两个场景的DataFrame
zh_df = df[df['Title'].isin(zh_titles)]
en_df = df[df['Title'].isin(en_titles)]
name=model_dir.split('/')[-1]
zh_df.to_csv(f'zh_model_detail_text_{name}_text_all_710.csv')
en_df.to_csv(f'en_model_detail_text_{name}_text_all_710.csv')


# 计算两种场景下各模型的指标平均值
zh_model_means = zh_df.groupby(['Judger','Narrator','Model'])[metrics].mean()
en_model_means = en_df.groupby(['Judger','Narrator','Model'])[metrics].mean()
zh_model_means['Average'] = zh_model_means[metrics].mean(axis=1)
en_model_means['Average'] = en_model_means[metrics].mean(axis=1)
zh_model_means=zh_model_means.sort_values(by='Average',ascending=False)
en_model_means=en_model_means.sort_values(by='Average',ascending=False)
# 输出结果查看
print("中文场景各模型指标平均值:")
print(zh_model_means)
print("\n英文场景各模型指标平均值:")
print(en_model_means)

# 如果需要，也可以将这些结果保存到CSV文件

zh_model_means.to_csv(f'zh_model_averages_text_{name}_text_all_710.csv')
en_model_means.to_csv(f'en_model_averages_text_{name}_text_all_710.csv')
    
preds=np.array(preds)
labels=np.array(labels)
rmse=np.sqrt(np.mean((preds-labels)**2))
print(rmse)
        #print(score)
    