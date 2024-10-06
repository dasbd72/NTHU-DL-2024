import logging
import os
import re
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Optional

import nltk
import pandas as pd
import textstat
from bs4 import BeautifulSoup
from dateutil import parser
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from tqdm import tqdm

nltk.download("stopwords", quiet=True)


def lowercase(x: str) -> str:
    return x.lower()


def remove_non_alphanumeric(x: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", "", x)


def remove_extra_whitespace(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


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
            return result


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
    DATA_VERSION = "0.1"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            self.logger.addHandler(stream_handler)

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.columns = self.get_columns()
        self.columns_to_extract = []

    def self_test(self):
        self.logger.info("Self test")
        self.logger.info("Extracting from test html")
        test_html = "<html><head><title>Test</title></head><body><h1>Test</h1><p>Test</p></body></html>"
        info = self.extract([test_html])
        assert info.shape == (1, len(self.columns))
        self.logger.info("Self test passed")

    def extract(
        self,
        arr: List[str],
        n_jobs: Optional[int] = None,
        cache_path: Optional[str] = None,
    ):
        self.logger.info("Extracting features")
        if cache_path is not None:
            if os.path.exists(cache_path):
                self.logger.info("Found cache, loading")
                df = pd.read_csv(cache_path)
            else:
                df = self._extract(arr, n_jobs=n_jobs)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path, index=False)
        else:
            df = self._extract(arr, n_jobs=n_jobs)
        return df

    def _extract(self, arr: List[str], n_jobs: Optional[int] = None):
        func = partial(self._extract_single)
        with Pool(n_jobs) as p:
            values = list(tqdm(p.imap(func, arr), total=len(arr)))
        df = pd.DataFrame(values, columns=self.columns)
        return df

    def _extract_single(self, html: str) -> List:
        soup = BeautifulSoup(html, "html.parser")
        info = []
        # Extract datetime
        info += self._extract_datetime(soup)
        # Extract channel
        info += self._extract_channel(soup)
        # Extract categories
        info += self._extract_categories(soup)
        # Extract by selector
        text_selectors = [
            "h1",
            "h2",
            "p",
            "a",
            "div",
            "p",
            "section",
            "footer>a[href]",
        ]
        non_text_selectors = [
            "img",
            "iframe",
            "video",
            ".instagram-media",
            ".twitter-tweet",
        ]
        for selector in text_selectors:
            info += self._extract_by_selector(
                soup, selector, analyze_text=True
            )
        for selector in non_text_selectors:
            info += self._extract_by_selector(
                soup, selector, analyze_text=False
            )
        return info

    def get_columns(self):
        column = []
        datetime_columns = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekday",
        ]
        column += datetime_columns

        channel_columns = ["channel"]
        column += channel_columns

        categories_columns = ["categories"]
        column += categories_columns

        text_selector_names = [
            "h1",
            "h2",
            "p",
            "a",
            "div",
            "p",
            "section",
            "footer_a",
        ]
        for name in text_selector_names:
            column += self._column_names_by_selector_name(
                name, analyze_text=True
            )
        non_text_selector_names = [
            "img",
            "iframe",
            "video",
            "instagram",
            "twitter",
        ]
        for name in non_text_selector_names:
            column += self._column_names_by_selector_name(
                name, analyze_text=False
            )
        return column

    def _extract_datetime(self, soup: BeautifulSoup) -> List:
        # Possible datetime selectors
        time_elements = soup.find_all("time", {"datetime": True})
        if time_elements is None:
            return None
        dt = None
        for time_element in time_elements:
            dt = parser.parse(time_element["datetime"])
        # Extract datetime
        if dt:
            return [
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                dt.weekday(),
            ]
        return [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

    def _extract_channel(self, soup: BeautifulSoup) -> List[str]:
        channel = soup.find("article", {"data-channel": True})
        if not channel:
            return [None]
        return [channel["data-channel"]]

    def _extract_categories(self, soup: BeautifulSoup) -> List[str]:
        # Find the footer containing the categories
        footer = soup.find("footer", class_="article-topics")
        if footer is None:
            return [""]
        # Extract the categories (anchor text) into a list of strings
        categories_set = set()
        for a in footer.find_all("a", href=True):
            category = a["href"].strip("/category/").strip("/").lower()
            category = re.sub(r"[^a-zA-Z\s]", "", category)
            if category == "":
                continue
            categories_set.add(category)
        categories = " ".join(list(categories_set))
        return [categories]

    def _extract_by_selector(
        self,
        soup: BeautifulSoup,
        selector: str,
        analyze_text: bool = False,
    ) -> List[int]:
        # Extract elements by selector
        el = soup.select(selector)
        info = []
        # Count of elements
        info += [len(el)]
        if not analyze_text:
            return info

        # Text of elements
        text = " ".join([e.get_text() for e in el])
        text = lowercase(text)
        text = remove_non_alphanumeric(text)
        text = remove_extra_whitespace(text)
        # Token count
        info += [len(text.split())]
        # Unique token count
        info += [len(set(text.split()))]
        # Non-stopword token count
        non_stopword_tokens = [
            t for t in text.split() if t not in stopwords.words("english")
        ]
        info += [len(non_stopword_tokens)]
        # Non-stopword unique token count
        info += [len(set(non_stopword_tokens))]
        # Sentiment
        sentiment = TextBlob(text).sentiment
        info += [sentiment.polarity, sentiment.subjectivity]
        # Readability
        info += [textstat.flesch_reading_ease(text)]
        return info

    def _column_names_by_selector_name(
        self, name: str, analyze_text: bool = False
    ) -> List[str]:
        columns = [
            f"{name}_count",
        ]
        if not analyze_text:
            return columns
        columns += [
            f"{name}_token_count",
            f"{name}_unique_token_count",
            f"{name}_non_stop_token_count",
            f"{name}_non_stop_unique_token_count",
            f"{name}_polarity",
            f"{name}_subjectivity",
            f"{name}_readability",
        ]
        return columns
