import json
file_path="/home/v-leiwang8/sft_july/LLaMA-Factory/data/reward/evaluation_rm_validity.json"
with open(file_path,'r',encoding='utf-8') as f:
    data=json.load(f)

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
instruct="First, provide a critique of the performance. Based on this critique, assign a score."
for i in range(0, len(data)):
   # batch_prompts = data[i:i+batch_size]['instruction']
    
    original_string=data[i]['instruction']
    index = original_string.find("\n<Scene>")

    modified_string = original_string[:index] + instruct + original_string[index:]
    modified_string=modified_string.replace("\n<Score>:\n","")
    #prompt=data[j]['instruction']+critic
    data[i]['instruction']=modified_string

with open("evaluation_critic_human.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)