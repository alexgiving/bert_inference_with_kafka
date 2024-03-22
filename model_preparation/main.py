from pathlib import Path
from typing import Tuple

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def prepare_model_tokenizer(model_name: str, num_classes: int, ckpt_path: Path
    ) -> Tuple[BertForSequenceClassification, BertTokenizerFast]:

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    if ckpt_path:
        model = BertForSequenceClassification.from_pretrained(ckpt_path)
    else:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    for param in model.bert.parameters():
        param.requires_grad = False
    return model, tokenizer


accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main() -> None:
    checkpoint_path = Path('data/checkpoints')
    dataset_name = 'ag_news'
    model_name = 'distilbert/distilbert-base-uncased'

    dataset = load_dataset(dataset_name)
    num_classes = 4
    model, tokenizer = prepare_model_tokenizer(model_name, num_classes, None)

    preprocess_function = lambda x: tokenizer(x['text'], truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy = 'epoch',
        load_best_model_at_end=True,
        output_dir=checkpoint_path
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(checkpoint_path / 'best')
    return


if __name__ == "__main__":
    main()
