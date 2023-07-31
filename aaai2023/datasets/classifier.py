
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Sample:
    id: str
    text: str
    label: bool

    def json(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
        }

    @staticmethod
    def from_json(d: dict) -> 'Sample':
        return Sample(**d)


@dataclass(frozen=True)
class TrainDataset:
    name: str
    train_samples: List[Sample]
    dev_samples: Optional[List[Sample]] = None

    def json(self) -> dict:
        return {
            "name": self.name,
            "train_samples": [s.json() for s in self.train_samples],
            "dev_samples": [s.json() for s in self.dev_samples] if self.dev_samples is not None else None,
        }

    @staticmethod
    def from_json(d: dict) -> 'TrainDataset':
        return TrainDataset(
            name=d['name'],
            train_samples=[Sample.from_json(s) for s in d['train_samples']],
            dev_samples=[Sample.from_json(s) for s in d['dev_samples']] if d['dev_samples'] is not None else None,
        )


@dataclass(frozen=True)
class TestDataset:
    name: str
    test_samples: List[Sample]

    def json(self) -> dict:
        return {
            "name": self.name,
            "test_samples": [s.json() for s in self.test_samples],
        }

    @staticmethod
    def from_json(d: dict) -> 'TestDataset':
        return TestDataset(
            name=d['name'],
            test_samples=[Sample.from_json(s) for s in d['test_samples']],
        )
