import os
import sys
import logging
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import evaluate
import wandb

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("/kaggle/input/trainset/PromiseEval_Trainset_English_1737261541086.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/dataset-tsv/testset_parquat2json_1737261439864.tsv", header=0, delimiter="\t", quoting=3)


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                    F.softmax(target, dtype=torch.float32), reduction=reduction)
    return loss


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        kl_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        kl_output = kl_outputs[1]
        kl_output = self.dropout(kl_output)
        kl_logits = self.classifier(kl_output)

        loss = None
        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            ce_loss = loss_fct(kl_logits.view(-1, self.num_labels), labels.view(-1))
            kl_loss = (KL(logits, kl_logits, "sum") + KL(kl_logits, logits, "sum")) / 2.
            total_loss = loss + ce_loss + kl_loss

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

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
    # promise_status_encoder = LabelEncoder()
    # evidence_status_encoder = LabelEncoder()
    verification_timeline_encoder = LabelEncoder()
    
    # 转换训练集标签,验证集和测试集标签
    verification_timeline_encoder.fit(train["verification_timeline"])
    print("train Classes:", verification_timeline_encoder.classes_)
    train["verification_timeline"] = verification_timeline_encoder.transform(train["verification_timeline"])  
    val["verification_timeline"] = verification_timeline_encoder.transform(val["verification_timeline"])
    
    train_dict = {
        # 'label': train["promise_status"],  
        'label': train["verification_timeline"],
        # 'label': train["evidence_status"],
        # 'label': train["evidence_quality"],
        'text': train['data']
    }
    val_dict = {
        # 'label': val["promise_status"],  
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

    # model = BertScratch.from_pretrained('bert-base-uncased', num_labels=2)
    model = BertScratch.from_pretrained(
        'bert-base-uncased',
        num_labels=5,
    )

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
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

    print(tokenized_train[0])  # 检查第一个样本的内容
    print(tokenized_val[0])    # 检查第一个样本的内容

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"ID": test["ID"], "verification_timeline": test_pred})
    result_output.to_csv("/kaggle/working/evidence_status.csv", index=False, quoting=3)
    logging.info('result saved!')
