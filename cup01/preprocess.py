import re
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Optional

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)


class Preprocessor(object):
    def __init__(self):
        self.pipeline = []

    def add_func(self, func: Callable[[str], str]):
        self.pipeline.append(func)
        return self

    def _parse_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

    def add_parse_html(self):
        self.add_func(partial(self._parse_html))
        return self

    def _parse_html_tags(self, html: str, tags: List[str]) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return " ".join([tag.get_text() for tag in soup.find_all(tags)])

    def add_parse_html_tags(self, tags: List[str]):
        self.add_func(partial(self._parse_html_tags, tags=tags))
        return self

    def _lowercase(self, x: str) -> str:
        return x.lower()

    def add_lowercase(self):
        self.add_func(partial(self._lowercase))
        return self

    def _remove_sentences(self, x: str) -> str:
        # Remove 'SEE ALSO' and similar patterns
        return re.sub(r"see also:.*", "", x)

    def add_remove_sentences(self):
        self.add_func(partial(self._remove_sentences))
        return self

    def _remove_whitespace(self, x: str) -> str:
        return re.sub(r"\s+", " ", x).strip()

    def add_remove_whitespace(self):
        self.add_func(partial(self._remove_whitespace))
        return self

    def _remove_non_alphanumeric(self, x: str) -> str:
        return re.sub(r"[^a-zA-Z\s]", "", x)

    def add_remove_non_alphanumeric(self):
        self.add_func(partial(self._remove_non_alphanumeric))
        return self

    def add_preprocess_text(self):
        self.add_lowercase()
        self.add_remove_sentences()
        self.add_remove_whitespace()
        self.add_remove_non_alphanumeric()
        return self

    def process_one(self, x: str) -> str:
        for func in self.pipeline:
            x = func(x)
        return x

    def process(
        self, arr: List[str], n_jobs: Optional[int] = None
    ) -> List[str]:
        func = partial(self.process_one)
        with Pool(n_jobs) as p:
            result = p.map(func, arr)
            return p.map(func, arr)


class Tokenizer(object):
    def __init__(self):
        self.pipeline = []
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def add_func(self, func: Callable[[List[str]], List[str]]):
        self.pipeline.append(func)
        return self

    def _no_stop_words(self, arr: List[str]) -> List[str]:
        return [token for token in arr if token not in self.stop_words]

    def add_no_stop_words(self):
        self.add_func(partial(self._no_stop_words))
        return self

    def _stemming(self, arr: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in arr]

    def add_stemming(self):
        self.add_func(partial(self._stemming))
        return self

    def process_one(self, x: str) -> List[str]:
        x = re.split(r"\s+", x.strip())
        for func in self.pipeline:
            x = func(x)
        return x

    def process(
        self, arr: List[str], n_jobs: Optional[int] = None
    ) -> List[List[str]]:
        func = partial(self.process_one)
        with Pool(n_jobs) as p:
            return p.map(func, arr)
