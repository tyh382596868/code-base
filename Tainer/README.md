# How to use Trainer of transformers library

## Installation

```
#!/bin/bash 
shopt -s expand_aliases
source ~/.bashrc
proxy_on


conda create -n py310 -y python=3.10
conda activate py310

proxy_off
pip install transformers==4.57.0 datasets torch==2.2.0 torchvision==0.17.0 -i https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/ --trusted-host pkg.pjlab.org.cn

```

## Model Checkpoints



## Running Inference for a Pre-Trained Model


## Fine-Tuning Base Models on Your Own Data

### 1. Download Data and Process with Tokenizer

```python
from datasets import load_dataset
from transformers import AutoTokenizer

## huggingface load_dataset加载数据集
dataset = load_dataset("Yelp/yelp_review_full")
# dataset["train"].to_pandas() #转成pandas看一下数据

## 对数据进行tokenize处理，map时启动batched大概速到要快上三倍
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length",truncation=True)

dataset = dataset.map(tokenize, batched=True)
```
### 2.Load Model
```python 
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5) # 数据集label数量是5，可以从数据集dataset card里看到
```

### 3.Training Model 
```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    eval_strategy="epoch",
    save_steps=5000
    # num_train_epochs=10,
)
print(training_args.num_train_epochs)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()

```

## Evalute Your Trained Model
```sh 
python eval.py --model_dir ./yelp_review_classificer/checkpoint-243750

```

## result

| Checkpoint Epoch   | Accuracy        |
| ------------------ | --------------- |
| 10000              | 0.58922         |
| 50000              | 0.63452         |
| 100000             | 0.27962         |
| 150000             | 0.64994         |
| 200000             | 0.65972         |
| 243750             | 0.6662          |
