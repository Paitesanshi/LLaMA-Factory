import json
import csv
import re
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

record=[]
with open('/home/v-leiwang8/sft_july/LLaMA-Factory/data/reward/evaluation_critic_short_new_test.json') as json_file:
    gpt_datas = json.load(json_file)


for i,data in enumerate(gpt_datas):
    labels=extract_scores(data['output'])
    temp = {'id':i,'Title': data['title'], 'Judger': data['judger'], 'Narrator': data['narrator'], 'Model': data['model']}
    criteria = ['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
    for j, criterion in enumerate(criteria):
        temp[criterion] = labels[j]
    temp['Average'] = sum(labels) / len(labels)
    record.append(temp)

with open('gpt_result.json', 'w') as f:
    json.dump(record, f, indent=4)

