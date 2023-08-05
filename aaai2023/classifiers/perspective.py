
from pathlib import Path
import shelve

import requests


class Perspective:

    class PerspectiveException(Exception):

        def __init__(self, status_code: int, text: str):
            body = f"status code: {status_code}\n\n{text}"
            super().__init__(body)

    def __init__(self, api_key: str, cache_file: Path):
        self.__api_key = api_key
        self.cache_file = cache_file
        self.db = None

        self.languages = ["en"]
        self.attributes = {
            "TOXICITY": {},
            "SEVERE_TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {},
            "PROFANITY": {},
            "THREAT": {},
        }
        self.url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    def __enter__(self):
        if not self.cache_file.exists():
            self.db = shelve.open(str(self.cache_file))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
        self.db = None

    def request(self, text: str):

        body = {
            "comment": {"text": text},
            "languages": self.languages,
            "requestedAttributes": self.attributes,
        }
        req = requests.post(
            url=self.url,
            json=body,
            params={"key": self.__api_key},
        )

        if req.status_code != 200:
            raise Perspective.PerspectiveException(
                status_code=req.status_code,
                text=req.text,
            )

        return req.json()

    def __call__(
        self,
        sample_key: str,
        sample_txt: str,
    ):
        res = self.db.get(sample_key)
        if res is None:
            res = self.request(sample_txt)
            self.db[sample_key] = res
        return res
