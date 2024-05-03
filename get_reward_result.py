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


from models.glm3_reward.modeling_chatglm import ChatGLMRM
from models.glm3_reward.tokenization_chatglm import ChatGLMTokenizer
# model_dir="/home/v-leiwang8/ChatGLM3/finetune_demo/output/lora_voldemort/checkpoint-8000"
model_dir="/home/v-leiwang8/LLaMA-Factory/models/glm3_reward_10k"
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = ChatGLMRM.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
print(model)
file_path="/home/v-leiwang8/RoleAgent/output/data/reward/evaluation_detail.json"
with open(file_path,'r',encoding='utf-8') as f:
    data=json.load(f)

csv_path="reward_result.csv"
fieldnames = ['Title','Judger','Narrator','Model', 'Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence','Average']
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

#record=pd.DataFrame(columns=fieldnames)

zh_titles=['西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖']
en_titles=['Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]
records=[]
#zh_record
#en_record={}
preds=[]
lables=[]
for i in tqdm(range(len(data))):
    prompt=data[i]['instruction']
    input_ids = tokenizer.encode(text=prompt, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(input_ids) > 8192:
        input_ids = input_ids[-8192:]
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    with torch.no_grad():
        score = model(input_ids=input_ids)[2].item() * 4 + 1
        preds.append(score)
        lables.append(data[i]['output'])
    criterion=data[i]['criteria']
    temp={'Title':data[i]['title'],'Judger':data[i]['judger'],'Narrator':data[i]['narrator'],'Model':data[i]['model'],'Knowledge Accuracy':0,'Emotional Expression':0,'Personality Traits':0,'Behavioral Accuracy':0,'Immersion':0,'Adaptability':0,'Behavioral Coherence':0,'Average':0}
    temp[criterion]=score
    records.append(temp)


df = pd.DataFrame(records)

# 为每条记录计算评估指标的平均值
metrics = ['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
df[metrics] = df[metrics] * 7
df['Average'] = df[metrics].mean(axis=1)

# 创建两个场景的DataFrame
zh_df = df[df['Title'].isin(zh_titles)]
en_df = df[df['Title'].isin(en_titles)]

# 计算两种场景下各模型的指标平均值
zh_model_means = zh_df.groupby(['Judger','Narrator','Model'])[metrics].mean()
en_model_means = en_df.groupby(['Judger','Narrator','Model'])[metrics].mean()
zh_model_means['Average'] = zh_model_means[metrics].mean(axis=1)
en_model_means['Average'] = en_model_means[metrics].mean(axis=1)
# 输出结果查看
print("中文场景各模型指标平均值:")
print(zh_model_means)
print("\n英文场景各模型指标平均值:")
print(en_model_means)

# 如果需要，也可以将这些结果保存到CSV文件
zh_model_means.to_csv('zh_model_averages.csv')
en_model_means.to_csv('en_model_averages.csv')
    
preds=np.array(preds)
lables=np.array(lables)
rmse=np.sqrt(np.mean((preds-lables)**2))
print(rmse)
        #print(score)
    