# Fine-Tuning Assistant Model with Yahoo Answers Data

This Python script demonstrates how to prepare and save data for fine-tuning an assistant model using the Yahoo Answers dataset. The script loads a sample of the dataset, formats it into a suitable structure, and splits it into training and validation sets. Finally, it saves the data into JSONL files for further use.

## Table of Contents
- [Description](#description)
- [Dependencies](#dependencies)
- [Code Explanation](#code-explanation)
- [Usage](#usage)
- [Example](#example)
- [Conclusion](#conclusion)

## Description

This project focuses on preparing data for training and validating a conversational model. It uses the Yahoo Answers dataset, formats the data for training, and saves it in JSONL format. The goal is to create a dataset suitable for fine-tuning an AI model to handle user questions with a polite and helpful tone.

## Dependencies

- Python 3.x
- openai library (for FineTuningJob and ChatCompletion)
- datasets library (for loading datasets)
- json (standard Python library)
- random and time (standard Python libraries)

You can install the required libraries with:
```
pip install openai datasets
```
## Code Explanation

### 1. Load Dataset
```
from datasets import load_dataset

yahoo_answers_qa = load_dataset("yahoo_answers_qa", split="train")
```
- Loads the Yahoo Answers dataset. The split="train" parameter specifies that we are using the training split of the dataset.

### 2. Sample Data
```
SAMPLE_SIZE = 150
yahoo_answers_qa = yahoo_answers_qa.select(range(SAMPLE_SIZE))
```
- Selects a subset of the data for processing. In this case, it takes the first 150 samples from the dataset.

### 3. Format Data
```
def format_data(data):
    formatted_data = [{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer users' question with a polite tone"},
            {"role": "user", "content": message["question"]},
            {"role": "assistant", "content": message["answer"]}
        ] 
    } for message in data]
    
    random.shuffle(formatted_data)
    
    return formatted_data

formatted_data = format_data(yahoo_answers_qa)
```
- Defines the format_data function that formats each data entry into a structured format suitable for model training. The function:
  - Creates a list of dictionaries with roles ("system", "user", "assistant") and corresponding content.
  - Shuffles the formatted data to ensure a diverse training set.

### 4. Split Data
```
TRAIN_SIZE = int(len(formatted_data) * 0.7)

training_data = formatted_data[:TRAIN_SIZE]
validation_data = formatted_data[TRAIN_SIZE:]
print(f"Training Data Size: {len(training_data)}")
print(f"Validation Data Size: {len(validation_data)}")
```
- Splits the formatted data into training and validation sets. Here, 70% of the data is used for training, and the remaining 30% is used for validation.

### 5. Save Data

```
def save_data(dictionary_data, file_name):
    with open(file_name, "w") as outfile:
        for entry in dictionary_data:
            json.dump(entry, outfile) 
            outfile.write("\n")

save_data(training_data, "training_data.jsonl")
save_data(validation_data, "validation_data.jsonl")
```
- Defines the save_data function that saves the data into JSONL files. Each line in the JSONL file contains a single JSON object.
- Saves the training and validation data into training_data.jsonl and validation_data.jsonl, respectively.

## Usage

1. Install Dependencies: Ensure you have Python 3.x and the necessary libraries installed.
2. Run the Script: Copy the code into a Python file (e.g., prepare_data.py).
3. Execute the Script: Run the script using Python:
 ``` 
   python prepare_data.py
 ```  
4. Check Outputs: Verify that the training_data.jsonl and validation_data.jsonl files have been created in the same directory as the script.

## Example
The script processes the Yahoo Answers dataset to create training and validation sets. The output files training_data.jsonl and validation_data.jsonl will be ready for use in fine-tuning a model.

## Conclusion

This script prepares and formats data from the Yahoo Answers dataset for fine-tuning a conversational AI model. It provides a clear and structured approach to data preparation, essential for training models to handle user interactions effectively.

