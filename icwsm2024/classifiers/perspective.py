
import requests


class Perspective:

    class PerspectiveException(Exception):

        def __init__(self, status_code: int, text: str):
            body = f"status code: {status_code}\n\n{text}"
            super().__init__(body)

    def __init__(self, api_key: str):
        self.__api_key = api_key

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

    def __call__(self, text: str):

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
