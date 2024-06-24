import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
import os
import re
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_prompt(row):
    return f"Caption: {row['Transcriptions']}\n" \
           f"Had you noticed you were feeling this way before we asked?: {int(row['ema_aware'])}\n" \
           f"Did you feel you were supported by others?: {int(row['ema_support'])}\n" \
           f"Did you recognize how your feelings were influencing your outlook on things?: {int(row['ema_insight'])}\n" \
           f"How fulfilled did you feel?: {int(row['ema_fulfilled'])}\n" \
           f"How hopeless did you feel?: {int(row['ema_hopeless'])}\n" \
           f"How anxious did you feel?: {int(row['ema_anxious'])}"
           
error_sums = {
    "aware": 0,
    "support": 0,
    "insight": 0,
    "fulfilled": 0,
    "hopeless": 0,
    "anxious": 0
}
num_iterations = 50

def make_prompt(df):
  sampled_rows = df.sample(n=4)
  example_rows = sampled_rows.iloc[:3]
  test_row = sampled_rows.iloc[3]

  prompt = "\n\n".join(example_rows.apply(format_prompt, axis=1))
  instructions = f"\n\nBased on the previous entries, predict the ratings for the following caption on a scale of 1 to 5. Provide your answers in the following JSON format and nothing else:\n" \
               "{\n" \
               "  \"aware\":,\n" \
               "  \"support\":,\n" \
               "  \"insight\":,\n" \
               "  \"fulfilled\":,\n" \
               "  \"hopeless\":,\n" \
               "  \"anxious\":\n" \
               "}\n"

  last_caption = test_row['Transcriptions']

  prompt += f"{instructions}\n\nCaption: {last_caption}\n" \
            "Had you noticed you were feeling this way before we asked?: \n" \
            "Did you feel you were supported by others?: \n" \
            "Did you recognize how your feelings were influencing your outlook on things?: \n" \
            "How fulfilled did you feel?: \n" \
            "How hopeless did you feel?: \n" \
            "How anxious did you feel?: \n"

  return prompt, test_row


def extract_valid_json(text):
    pattern = r'\{.*?\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    valid_json_objects = []

    for match in matches:
        json_str = match.group(0)

        if re.search(r'"aware":\s*\d+', json_str):
            try:
                json_data = json.loads(json_str)
                if isinstance(json_data.get("aware"), int):
                    return json_data
            except json.JSONDecodeError:
                continue

    return valid_json_objects


for _ in range(num_iterations):

    prompt = make_prompt(df)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(input_ids=inputs['input_ids'].to("cuda"), max_new_tokens=200)
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    predicted_values = extract_valid_json(predicted_text)

    for key in predicted_values.keys():
        error_sums[key] += abs(predicted_values[key] - int(test_row.iloc[0]['ema_' + key]))

average_errors = {key: value / num_iterations for key, value in error_sums.items()}

overall_average_error = np.mean(list(average_errors.values()))

print("Average errors for each attribute:", average_errors)
print("Overall average error:", overall_average_error)

import matplotlib.pyplot as plt

plt.bar(average_errors.keys(), average_errors.values(), color='skyblue')

plt.axhline(y=overall_average_error, color='red', linestyle='--', label=f'Overall Average Error: {overall_average_error:1f}')

plt.xlabel('Attributes')
plt.ylabel('Average Error')
plt.title('Average Errors for Each Attribute')

plt.legend()