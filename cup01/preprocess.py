import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Optional

import nltk
import numpy as np
import textstat
from bs4 import BeautifulSoup
from dateutil import parser
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob

nltk.download("stopwords", quiet=True)


class Preprocessor(object):
    def __init__(self):
        self.pipeline = []

    def add_func(self, func: Callable[[str], str]):
        self.pipeline.append(func)
        return self

    def _parse_html(self, html: str, no_head=True) -> str:
        soup = BeautifulSoup(html, "html.parser")
        if no_head:
            for tag in soup.find_all("head"):
                tag.decompose()
        return soup.get_text()

    def add_parse_html(self, no_head=True):
        self.add_func(partial(self._parse_html, no_head=no_head))
        return self

    def _parse_html_tags(
        self, html: str, tags: List[str], no_head=True
    ) -> str:
        soup = BeautifulSoup(html, "html.parser")
        if no_head:
            for tag in soup.find_all("head"):
                tag.decompose()
        return " ".join([tag.get_text() for tag in soup.find_all(tags)])

    def add_parse_html_tags(self, tags: List[str], no_head=True):
        self.add_func(
            partial(self._parse_html_tags, tags=tags, no_head=no_head)
        )
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


class Extractor(object):
    def __init__(self):
        pass

    def _try_extract_date_datetime(
        self, soup: BeautifulSoup
    ) -> datetime | None:
        # Possible datetime selectors
        time_elements = soup.find_all("time", {"datetime": True})
        if not time_elements:
            return None
        for time_element in time_elements:
            dt = parser.parse(time_element["datetime"])
            return dt
        return None

    def _extract_datetime(self, html: str) -> List:
        soup = BeautifulSoup(html, "html.parser")
        dt = self._try_extract_date_datetime(soup)
        if dt:
            return (
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                dt.weekday(),
            )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def extract_datetime(self, arr: List[str], n_jobs: Optional[int] = None):
        func = partial(self._extract_datetime)
        with Pool(n_jobs) as p:
            return np.array(p.map(func, arr))

    def datetime_columns(self):
        return [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekday",
        ]

    def _try_extract_channel(self, soup: BeautifulSoup) -> str | None:
        channel = soup.find("article", {"data-channel": True})
        if not channel:
            return None
        return channel["data-channel"]

    def _extract_channel(self, content):
        soup = BeautifulSoup(content, "html.parser")
        return self._try_extract_channel(soup)

    def extract_channel(self, arr: List[str], n_jobs: Optional[int] = None):
        func = partial(self._extract_channel)
        with Pool(n_jobs) as p:
            return np.array(p.map(func, arr))

    def _extract_counts(self, text: str) -> List[int]:
        soup = BeautifulSoup(text, "html.parser")
        selectors = [
            "h1",
            "h2",
            "img",
            "iframe",
            "video",
            ".instagram-media",
            ".twitter-tweet",
            "a",
            "div",
            "p",
            "section",
        ]
        # Counts of number of each selectors
        counts = [len(soup.select(selector)) for selector in selectors]
        # Words count of all text
        counts += [len(soup.get_text().split())]
        # Words count of all h1
        counts += [
            sum([len(h1.get_text().split()) for h1 in soup.select("h1")])
        ]
        return counts

    def extract_counts(self, texts: List[str]) -> np.ndarray:
        with Pool() as p:
            counts = p.map(self._extract_counts, texts)
        return np.array(counts)

    def counts_columns(self):
        return [
            "h1",
            "h2",
            "img",
            "iframe",
            "video",
            "instagram",
            "twitter",
            "a",
            "div",
            "p",
            "section",
            "wc",
            "wc_h1",
        ]

    def _extract_categories(self, html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        # Find the footer containing the categories
        footer = soup.find("footer", class_="article-topics")
        if footer is None:
            return []
        # Extract the categories (anchor text) into a list of strings
        categories_set = set()
        for a in footer.find_all("a", href=True):
            category = a["href"].strip("/category/").strip("/").lower()
            category = re.sub(r"[^a-zA-Z\s]", "", category)
            if category == "":
                continue
            categories_set.add(category)
        categories = np.array(list(categories_set))
        return categories

    def extract_categories(self, arr: List[str], n_jobs: Optional[int] = None):
        func = partial(self._extract_categories)
        with Pool(n_jobs) as p:
            return p.map(func, arr)

    def _extrain_scores(self, html: str) -> List[float]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        # Remove white spaces
        text = re.sub(r"\s+", " ", text).strip()
        # Remove non-alphanumeric characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        sentiment = TextBlob(text).sentiment
        scores = [
            sentiment.polarity,
            sentiment.subjectivity,
            textstat.flesch_reading_ease(text),
        ]
        return scores

    def extract_scores(self, arr: List[str], n_jobs: Optional[int] = None):
        func = partial(self._extrain_scores)
        with Pool(n_jobs) as p:
            return np.array(p.map(func, arr))

    def scores_columns(self):
        return ["polarity", "subjectivity", "readability"]
