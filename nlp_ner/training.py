from transformers import (
    BertTokenizerFast, 
    BertForTokenClassification, 
    DataCollatorForTokenClassification,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import numpy as np
import ast
import evaluate
from sklearn.model_selection import train_test_split
from functools import partial

# define tag to ID mapping
tag2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}
id2tag = {v: k for k, v in tag2id.items()}

# evaluation metric
metric = evaluate.load('seqeval')


def tokenize_and_align_labels(df, tokenizer):
    tokenized_inputs = tokenizer(df['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(df['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels.append([
            (tag2id[label[word_id]] if word_id is not None else -100)
            for word_id in word_ids
        ])
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
    }


def train_model(df):
    # load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2id))
    
    # preprocess data
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['tags'] = df['tags'].apply(ast.literal_eval)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenize_and_align = partial(tokenize_and_align_labels, tokenizer=tokenizer)
    tokenized_train_dataset = train_dataset.map(tokenize_and_align, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_and_align, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trained_model = trainer.model
    evaluation_results = trainer.evaluate()
    return trained_model, tokenizer, evaluation_results
