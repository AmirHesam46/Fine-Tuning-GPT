from openai import FineTuningJob, ChatCompletion
from datasets import load_dataset
from time import sleep
import random
import json

yahoo_answers_qa = load_dataset("yahoo_answers_qa", split="train")

SAMPLE_SIZE = 150
yahoo_answers_qa = yahoo_answers_qa.select(range(SAMPLE_SIZE))

def format_data(data):
    
    formatted_data = [{
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer users' question with a polite tone"},
                {"role": "user", "content": message["question"]},
                {"role": "assistant", "content": message["answer"]}
            ] 
        } for message in data 
    ]
    
    random.shuffle(formatted_data)
    
    return formatted_data

formatted_data = format_data(yahoo_answers_qa)

TRAIN_SIZE = int(len(formatted_data) * 0.7)

training_data = formatted_data[:TRAIN_SIZE]
validation_data = formatted_data[TRAIN_SIZE:]
print(f"Training Data Size: {len(training_data)}")
print(f"Validation Data Size: {len(validation_data)}")

def save_data(dictionary_data, file_name):
    
    with open(file_name, "w") as outfile:
        for entry in dictionary_data:
            json.dump(entry, outfile) 
            outfile.write("\n")

save_data(training_data, "training_data.jsonl")
save_data(validation_data, "validation_data.jsonl")

