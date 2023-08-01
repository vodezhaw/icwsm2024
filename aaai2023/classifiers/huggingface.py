
import gzip
import json
from pathlib import Path
from typing import List

import torch
from torch.nn.functional import softmax
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


from aaai2023.datasets.classifier import (
    Sample,
    ScoredSample,
    TrainDataset,
    TestDataset,
    ScoredDataset,
)


class TorchDataset(Dataset):

    def __init__(self, data: dict):
        self.data = data

    def __len__(self) -> int:
        return self.data['input_ids'].shape[0]

    def __getitem__(self, item: int):
        return {
            k: v[item]
            for k, v in self.data.items()
        }

    @staticmethod
    def from_samples(
        samples: List[Sample],
        tokenizer: AutoTokenizer,
        include_labels: bool = True,
    ) -> 'TorchDataset':
        torch_data = tokenizer(
            [s.text for s in samples],
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        if include_labels:
            torch_data['labels'] = torch.LongTensor([s.label for s in samples])

        return TorchDataset(data=torch_data)


def hf_eval(eval_pred):
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    y_true = eval_pred.label_ids

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def finetune(
    model_short_name: str,
    base_model: str,
    train_data: TrainDataset,
    test_data: List[TestDataset],
    scores_dir: Path,
    model_dir: Path,
    dev_mode: bool = False,
):
    if not model_dir.exists():
        model_dir.mkdir()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

    if train_data.dev_samples is not None:
        train_samples = train_data.train_samples
        dev_samples = train_data.dev_samples
    else:
        train_samples, dev_samples = train_test_split(
            train_data.train_samples,
            test_size=.1,
            random_state=0xdeadbeef,
            shuffle=True,
            stratify=[s.label for s in train_data.train_samples],
        )

    if dev_mode:
        train_samples = train_samples[:32]
        dev_samples = dev_samples[:32]

    train = TorchDataset.from_samples(
        samples=train_samples,
        tokenizer=tokenizer,
        include_labels=True,
    )
    dev = TorchDataset.from_samples(
        samples=dev_samples,
        tokenizer=tokenizer,
        include_labels=True,
    )

    hyper_params = {
        "epochs": 1 if dev_mode else 50,
        "batch_size": 16,
        "warmup_steps": 250,
        "weight_decay": .01,
        "learning_rate": 5e-5,
        "early_stopping_patience": 5,
    }

    training_output = model_dir / "training_output"
    if not training_output.exists():
        training_output.mkdir()
    args = TrainingArguments(
        output_dir=str(training_output),
        num_train_epochs=hyper_params['epochs'],
        per_device_train_batch_size=hyper_params['batch_size'],
        per_device_eval_batch_size=hyper_params['batch_size'],
        warmup_steps=hyper_params['warmup_steps'],
        weight_decay=hyper_params['weight_decay'],
        learning_rate=hyper_params['learning_rate'],
        save_strategy="epoch",
        save_total_limit=1,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=hf_eval,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hyper_params['early_stopping_patience'])],
    )
    trainer.train()

    trainer.save_model(output_dir=str(model_dir))
    tokenizer.save_pretrained(save_directory=str(model_dir))

    for test in test_data:
        test_samples = test.test_samples
        if dev_mode:
            test_samples = test_samples[:32]

        dataset = TorchDataset.from_samples(
            samples=test_samples,
            tokenizer=tokenizer,
            include_labels=False,
        )
        preds = trainer.predict(test_dataset=dataset)
        scores = softmax(
            torch.from_numpy(preds.predictions),
            dim=1,
        )[:, 1]

        scored_samples = [
            ScoredSample(
                id=sample.id,
                score=score.item(),
            )
            for sample, score in zip(test_samples, scores)
        ]

        out = ScoredDataset(
            classifier_name=base_model,
            train_data=train_data.name,
            test_data=test.name,
            classifier_params=hyper_params,
            default_threshold=.5,
            scores=scored_samples,
        )

        with gzip.open(scores_dir / f"{model_short_name}-{train_data.name}-{test.name}.json.gz", "wt") as fout:
            fout.write(json.dumps(obj=out.json(), indent=2))
            fout.write('\n')
