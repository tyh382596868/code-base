##################################################
# 准备数据

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

# small_train = dataset["train"].shuffle(seed=42).select(range(10)) # 选个子集线调试
# small_test = dataset["test"].shuffle(seed=42).select(range(10))

##################################################
# 加载模型

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5) # 数据集label数量是5，可以从数据集dataset card里看到

##################################################
# 训练模型

## 计算指标
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

## 加载训练参数
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    eval_strategy="epoch",
    save_steps=5000
    # num_train_epochs=10,
)
print(training_args.num_train_epochs)

## 加载训练器，开始训练
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.evaluate()


