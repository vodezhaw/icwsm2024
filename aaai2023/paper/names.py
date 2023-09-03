
TEST_DATASETS = [
    "davidson-hate-off",
    "davidson-hate-only",
    "dynamically-generated-hate",
    "ethos",
    "ex-machina",
    "hasoc2019",
    "hateval2019",
    "jigsaw",
    "olid",
    "solid",
    "swad",
    "wassa",
]

TRAIN_DATASETS = [
    "davidson-hate-off",
    "davidson-hate-only",
    "dynamically-generated-hate",
    "ex-machina",
    "hasoc2019",
    "hateval2019",
    "jigsaw",
    "olid",
    "wassa",
]

DATASET_NAMES = {
    "davidson-hate-off": "Davidson Hate & Offensive",
    "davidson-hate-only": "Davidson Hate Only",
    "dynamically-generated-hate": "DGH",
    "ethos": "Ethos",
    "ex-machina": "Ex Machina",
    "hasoc2019": "HASOC 2019",
    "hateval2019": "HatEval 2019",
    "jigsaw": "Jigsaw",
    "olid": "OLID",
    "solid": "SOLID",
    "swad": "SWAD",
    "wassa": "WASSA",
}

CLF = [
    "tfidf-svm",
    "google/electra-base-discriminator",
    "cardiffnlp/twitter-roberta-base",
    "perspective-api",
]

CLF_NAMES = {
    "tfidf-svm": "TF-IDF SVM",
    "google/electra-base-discriminator": "Electra",
    "cardiffnlp/twitter-roberta-base": "Twitter-RoBERTa",
    "perspective-api": "Perspective API",
}

CLF_FILE_STEMS = {
    "tfidf-svm": "tfidf-svm",
    "google/electra-base-discriminator": "electra",
    "cardiffnlp/twitter-roberta-base": "twitter-roberta",
    "perspective-api": "perspective",
}


QUANTIFICATION_STRATEGIES = [
    "CC",
    "ACC",
    "PCC",
    "PACC",
    "CPCC",
    "BCC",
]


SELECTION_STRATEGIES = [
    "random",
    "quantile",
]
