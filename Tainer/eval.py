from datasets import load_dataset
from transformers import AutoTokenizer

import argparse
from transformers import AutoModelForSequenceClassification

from transformers import TrainingArguments
from transformers import Trainer
import numpy as np 
import evaluate


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)

def get_model_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to the model checkpoint directory"
    )
    args = parser.parse_args()
    return args.model_dir

def main():
    ########################### data

    dataset = load_dataset("Yelp/yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    ########################### model
    model_dir = get_model_dir()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    ########################### Trainer


    training_args = TrainingArguments()
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['test'], compute_metrics=compute_metrics)
    result = trainer.evaluate()
    print(result)

    log_ebtry = {
        "model_dir": model_dir,
        "result": result
    }

    with open("eval_result.log", "a") as f:

        f.write(str(log_ebtry) + "\n")
    

if __name__=="__main__":
    main()