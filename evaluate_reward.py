from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from trl import AutoModelForCausalLMWithValueHead
import uvicorn, json, datetime
import torch
import os
from tqdm import tqdm
# model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")

DEVICE = "cuda"
DEVICE_ID = "8"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()
torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   max_length=max_length if max_length else 8196,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    #torch_gc()
    return answer


if __name__ == '__main__':
    model_dir = "THUDM/chatglm3-6b"
    adapter_path="/home/v-leiwang8/LLaMA-Factory/saves/ChatGLM/lora/point/checkpoint-1000"
    from models.glm3_reward.modeling_chatglm import ChatGLMRM
    from models.glm3_reward.tokenization_chatglm import ChatGLMTokenizer
    # model_dir="/home/v-leiwang8/ChatGLM3/finetune_demo/output/lora_voldemort/checkpoint-8000"
    tokenizer = ChatGLMTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = ChatGLMRM.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    print(model)
    file_path="/home/v-leiwang8/LLaMA-Factory/data/reward_data/evaluation.json"
    with open(file_path,'r') as f:
        data=json.load(f)
    
    ans=[]
    labels=[]
    num=100
    for i in tqdm(range(num)):
        labels.append(data[i]['output'])
        prompt=data[i]['instruction']
        input_ids = tokenizer.encode(text=prompt, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(input_ids) > 8192:
            input_ids = input_ids[-8192:]
        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
        with torch.no_grad():
            score = model(input_ids=input_ids)[2].item() * 4 + 1
            ans.append(score)
            #print(score)
    print(ans)
    rmse=(sum([(ans[i]-labels[i])**2 for i in range(num)])/num)**0.5
    print(rmse)
    #model.eval()

    # uvicorn.run(app, host='
    
    # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    # model = model.eval()

