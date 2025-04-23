import os
import sys
import logging
import datasets

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

import evaluate
import wandb
from sklearn.preprocessing import LabelEncoder

import optuna
from optuna.trial import Trial


train = pd.read_csv("/kaggle/input/trainset/PromiseEval_Trainset_English_1737261541086.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/dataset-tsv/testset_parquat2json_1737261439864.tsv", header=0, delimiter="\t", quoting=3)

# 定义目标函数
def objective(trial: Trial):
    # 超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000, step=100)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    
    # 数据预处理部分
    verification_timeline_encoder = LabelEncoder()
    verification_timeline_encoder.fit(train["promise_status"])
    print("train Classes:", verification_timeline_encoder.classes_)
    train["promise_status"] = verification_timeline_encoder.transform(train["promise_status"])  
    val["promise_status"] = verification_timeline_encoder.transform(val["promise_status"])
    
    train_dict = {'label': train["promise_status"], 'text': train['data']}
    val_dict = {'label': val["promise_status"], 'text': val['data']}
    test_dict = {"text": test['data']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",  
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,  # 限制保存的checkpoint数量
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 训练并返回验证集准确率
    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_accuracy"]

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)
    wandb.login(key="92f97c04cf55833aa5ca8fbc9253535126d9e0ee")

    # 创建Optuna study并优化
    study = optuna.create_study(direction="maximize")  
    study.optimize(objective, n_trials=10)  # 运行10次试验

    # 打印最佳试验结果
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("promise_status超参数如上")
