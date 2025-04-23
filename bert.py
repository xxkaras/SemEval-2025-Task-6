#只使用bert
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


train = pd.read_csv("/kaggle/input/trainset/PromiseEval_Trainset_English_1737261541086.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/dataset-tsv/testset_parquat2json_1737261439864.tsv", header=0, delimiter="\t", quoting=3)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    wandb.login(key="92f97c04cf55833aa5ca8fbc9253535126d9e0ee")


    print(train.columns)
    
    # 初始化LabelEncoder
    verification_timeline_encoder = LabelEncoder()
    
    # 转换训练集标签,验证集和测试集标签
    verification_timeline_encoder.fit(train["verification_timeline"])
    print("train Classes:", verification_timeline_encoder.classes_)
    train["verification_timeline"] = verification_timeline_encoder.transform(train["verification_timeline"])  
    val["verification_timeline"] = verification_timeline_encoder.transform(val["verification_timeline"])
    
    train_dict = {
        # 'label': train["promise_status"],  # 确保列名正确
        'label': train["verification_timeline"],
        # 'label': train["evidence_status"],
        # 'label': train["evidence_quality"],
        'text': train['data']
    }
    val_dict = {
        # 'label': val["promise_status"],  # 确保列名正确
        'label': val["verification_timeline"],
        # 'label': val["evidence_status"],
        # 'label': val["evidence_quality"],
        'text': val['data']
    }
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

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=5)

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=4,  #16 batch size per device during training
        per_device_eval_batch_size=8,  #32 batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.03,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"ID": test["ID"], "verification_timeline": test_pred})
    result_output.to_csv("/kaggle/working/bert_evidence_quality.csv", index=False, quoting=3)
    logging.info('result saved!')
