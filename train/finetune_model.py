import pickle
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

DATA_DIR = "../data/generated-synthetic"

def load_pickle(filename: str):
  with open(filename, 'rb') as file:
    return pickle.load(file)

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return result


def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   model_inputs = tokenizer([ex for ex in examples['text']], padding=True, truncation=True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer([ex for ex in examples['label']], 
                      max_length=512,         
                      truncation=True)
    # reset the label to be tokenized, rather than original text label
   model_inputs["label"] = labels["input_ids"]
   return model_inputs

if __name__ == "__main__":
    # if you want to cache the model weights somewhere, you can specify that here
    cache_dir = "models/"
    
    MODEL_NAME = "google/flan-t5-small" # "google/flan-t5-base" #google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    model_type = "encoder_decoder"

    # Load data for finetuning, filter by 'include' key
    train_ds_path = DATA_DIR+'/updated_synthetic_train_train_op21970'
    ds_train = load_pickle(train_ds_path)
    fds_train = ds_train.filter(lambda example: example["include"])
    print(fds_train)
    # tokenize the data and labels
    tokenized_dataset = fds_train.map(preprocess_function, batched=True)
    # split the train, test data
    fds_train_split = tokenized_dataset.train_test_split(test_size=0.3)
    print(fds_train_split)
    


    # Based on paper, expect similar paramters to 8B Flan-PaLM 32 : 
    # Dropout, LR, Steps: 0.05 3×10−3 1k
    
    nltk.download("punkt", quiet=True)
    metric = evaluate.load("rouge")
    # Global Parameters
    L_RATE = 3e-3
    BATCH_SIZE = 32
    PER_DEVICE_EVAL_BATCH = 16
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIM = 3
    NUM_EPOCHS = 3

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./finetune-results",
        evaluation_strategy="epoch",
        learning_rate=L_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIM,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        push_to_hub=False
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=fds_train_split["train"],
        eval_dataset=fds_train_split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()