import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON files
with open('glm3_result.json', 'r') as file:
    glm3_data = json.load(file)

with open('gpt_result.json', 'r') as file:
    gpt_data = json.load(file)

with open('glm4_result.json', 'r') as file:
    glm4_data = json.load(file)

# Convert the data to DataFrames
glm3_df = pd.DataFrame(glm3_data)
gpt_df = pd.DataFrame(gpt_data)
glm4_df = pd.DataFrame(glm4_data)

# Extract the 7 individual scores for each model
glm3_scores = glm3_df[['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']]
gpt_scores = gpt_df[['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']]
glm4_scores = glm4_df[['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']]

# List of metrics
metrics = ['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']

# Create individual plots for each metric's score distribution
for metric in metrics:
    plt.figure(figsize=(10, 6))
    bins = np.arange(1, 7)
    glm3_hist, _ = np.histogram(glm3_scores[metric], bins=bins)
    gpt_hist, _ = np.histogram(gpt_scores[metric], bins=bins)
    glm4_hist, _ = np.histogram(glm4_scores[metric], bins=bins)
    
    total = glm3_hist.sum() + gpt_hist.sum() + glm4_hist.sum()
    
    bar_width = 0.2
    index = np.arange(1, 6)
    
    plt.bar(index - bar_width, glm3_hist / total * 100, width=bar_width, label='Reward Model(Value)', alpha=0.7)
    plt.bar(index, gpt_hist / total * 100, width=bar_width, label='GPT-4', alpha=0.7)
    plt.bar(index + bar_width, glm4_hist / total * 100, width=bar_width, label='Reward Model(Text)', alpha=0.7)
    
    plt.title(f'Score Distribution for {metric}', fontsize=15)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(index, [1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f'figs/{metric}_distribution.png')
    plt.show()
