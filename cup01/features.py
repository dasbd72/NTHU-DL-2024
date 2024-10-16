import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

import preprocess
from utils import do_or_load


class Features(object):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        extractor: Optional[preprocess.Extractor] = None,
        onehot_weekday: bool = False,
        onehot_month: bool = False,
        category_vectorize=False,
        category_max_features: Optional[int] = 1000,
        category_vectorizer=None,
        category_train_min: int = 1,
        category_test_min: int = 1,
        train_parsed_html_path: Optional[str] = None,
        test_parsed_html_path: Optional[str] = None,
        preprocessor: Optional[preprocess.Preprocessor] = None,
        train_tokens_path: Optional[str] = None,
        test_tokens_path: Optional[str] = None,
        tokenizer: Optional[preprocess.Tokenizer] = None,
    ):
        # Load data
        self.df_train = self.load(train_path)
        self.df_test = self.load(test_path)
        X_contents = self.df_train["Page content"].values
        y = np.where(self.df_train["Popularity"].values > 0, 1, 0)
        X_test_contents = self.df_test["Page content"].values

        # Input data
        self.X_contents = X_contents
        if y is not None:
            self.y = np.where(y > 0, 1, 0)
        else:
            self.y = y
        self.X_test_contents = X_test_contents

        # Extracting features from contents
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = preprocess.Extractor()
        self.onehot_weekday = onehot_weekday
        self.onehot_month = onehot_month
        self.category_vectorize = category_vectorize
        self.category_max_features = category_max_features
        if (
            not isinstance(category_vectorizer, TfidfVectorizer)
            and not isinstance(category_vectorizer, CountVectorizer)
            and category_vectorizer is not None
        ):
            raise ValueError("category_vectorizer must be a vectorizer")
        elif category_vectorizer is None:
            category_vectorizer = TfidfVectorizer(
                max_features=self.category_max_features
            )
        self.category_vectorizer = category_vectorizer
        self.category_train_min = category_train_min
        self.category_test_min = category_test_min
        self.X_info_raw: Optional[pd.DataFrame] = None
        self.X_info: Optional[pd.DataFrame] = None
        self.X_test_info_raw: Optional[pd.DataFrame] = None
        self.X_test_info: Optional[pd.DataFrame] = None

        # Extracting features from parsed html
        self.X_parsed_html: Optional[np.ndarray] = None
        self.X_test_parsed_html: Optional[np.ndarray] = None
        self.X_tokens: Optional[np.ndarray] = None
        self.X_test_tokens: Optional[np.ndarray] = None
        self.setup_extract_tokens(
            train_parsed_html_path,
            test_parsed_html_path,
            preprocessor,
            train_tokens_path,
            test_tokens_path,
            tokenizer,
        )

    def load(self, path: str):
        if not os.path.exists(path):
            raise ValueError("file path does not exist")
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("file path must be parquet or csv")

        return df

    def extract_info(self):
        self.X_info_raw = self.extractor.extract(
            self.X_contents,
            cache_path="./cache/train_extractor_cache_{}.parquet".format(
                preprocess.Extractor.DATA_VERSION
            ),
        )
        self.X_info = self.X_info_raw.copy(True)
        self.X_test_info_raw = self.extractor.extract(
            self.X_test_contents,
            cache_path="./cache/test_extractor_cache_{}.parquet".format(
                preprocess.Extractor.DATA_VERSION
            ),
        )
        self.X_test_info = self.X_test_info_raw.copy(True)

        if self.onehot_weekday:
            # One-hot encode weekday
            weekday_encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            weekday_encoder.fit([[i] for i in range(7)])

            def process_weekday(df: pd.DataFrame) -> pd.DataFrame:
                columns = ["datetime_weekday_{}".format(i) for i in range(7)]
                onehot = weekday_encoder.transform(
                    df["datetime_weekday"].values.reshape(-1, 1)
                )
                new_df = df.drop(columns=["datetime_weekday"])
                for i, column in enumerate(columns):
                    new_df[column] = onehot[:, i]
                return new_df

            self.X_info = process_weekday(self.X_info)
            self.X_test_info = process_weekday(self.X_test_info)

        if self.onehot_month:
            # One-hot encode month
            month_encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            month_encoder.fit([[i] for i in range(12)])

            def process_month(df: pd.DataFrame) -> pd.DataFrame:
                columns = ["datetime_month_{}".format(i) for i in range(12)]
                onehot = month_encoder.transform(
                    df["datetime_month"].values.reshape(-1, 1)
                )
                new_df = df.drop(columns=["datetime_month"])
                for i, column in enumerate(columns):
                    new_df[column] = onehot[:, i]
                return new_df

            self.X_info = process_month(self.X_info)
            self.X_test_info = process_month(self.X_test_info)

        # One-hot encode channel
        common_channels = set(self.X_info["channel"].unique()).intersection(
            set(self.X_test_info["channel"].unique())
        )
        channel_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        channel_encoder.fit([[channel] for channel in common_channels])

        def process_channel(df: pd.DataFrame) -> pd.DataFrame:
            columns = [
                "channel_{}".format(channel)
                for channel in channel_encoder.categories_[0]
            ]
            onehot = channel_encoder.transform(
                df["channel"].values.reshape(-1, 1)
            )
            new_df = df.drop(columns=["channel"])
            for i, column in enumerate(columns):
                new_df[column] = onehot[:, i]
            return new_df

        self.X_info = process_channel(self.X_info)
        self.X_test_info = process_channel(self.X_test_info)

        if self.category_vectorize:
            # Vectorize categories
            self.X_info["categories"].fillna("", inplace=True)
            self.X_test_info["categories"].fillna("", inplace=True)
            self.category_vectorizer.fit(self.X_info["categories"].values)

            def process_categories(df: pd.DataFrame) -> pd.DataFrame:
                columns = [
                    "category_{}".format(column)
                    for column in self.category_vectorizer.get_feature_names_out()
                ]
                new_df = df.drop(columns=["categories"])
                multihot = self.category_vectorizer.transform(
                    df["categories"].values
                ).toarray()
                for i, column in enumerate(columns):
                    new_df[column] = multihot[:, i]
                return new_df

            self.X_info = process_categories(self.X_info)
            self.X_test_info = process_categories(self.X_test_info)
        else:
            # Mulihot encode categories
            self.X_info["categories"].fillna("", inplace=True)
            self.X_test_info["categories"].fillna("", inplace=True)

            X_categories_list_str = [
                categories.split(" ")
                for categories in self.X_info["categories"]
            ]
            X_test_categories_list_str = [
                categories.split(" ")
                for categories in self.X_test_info["categories"]
            ]

            # Step 1: Count categories
            def count_categories(categories_list_str):
                categories_dict = {}
                for categories in categories_list_str:
                    for category in categories:
                        categories_dict[category] = (
                            categories_dict.get(category, 0) + 1
                        )
                return categories_dict

            all_X_categories_dict = count_categories(X_categories_list_str)
            all_X_test_categories_dict = count_categories(
                X_test_categories_list_str
            )

            # Step 2: Find common categories
            all_X_categories_set = set(all_X_categories_dict.keys())
            all_X_test_categories_set = set(all_X_test_categories_dict.keys())
            all_common_categories_set = all_X_categories_set.intersection(
                all_X_test_categories_set
            )

            # Step 3: Filter common categories with count > categories_min_count
            common_categories_dict = {}
            for category in all_common_categories_set:
                if (
                    all_X_categories_dict.get(category, 0)
                    >= self.category_train_min
                    and all_X_test_categories_dict.get(category, 0)
                    >= self.category_test_min
                ):
                    common_categories_dict[category] = (
                        all_X_categories_dict.get(category, 0)
                        + all_X_test_categories_dict.get(category, 0)
                    )
            common_categories_set = set(common_categories_dict.keys())
            common_categories = np.array(list(common_categories_set)).astype(
                str
            )

            # Step 4: Transform categories to one-hot
            mlb_categories = MultiLabelBinarizer(sparse_output=False)
            mlb_categories.fit(common_categories.reshape(-1, 1))

            self.categories_columns = common_categories
            self.categories_counts = np.array(
                [
                    common_categories_dict[category]
                    for category in common_categories
                ]
            )

            def process_categories(
                df: pd.DataFrame, categories_list_str: list[list[str]]
            ) -> pd.DataFrame:
                columns = [
                    "category_{}".format(column)
                    for column in self.categories_columns
                ]
                new_df = df.drop(columns=["categories"])
                multihot = mlb_categories.transform(categories_list_str)
                for i, column in enumerate(columns):
                    new_df[column] = multihot[:, i]
                return new_df

            self.X_info = process_categories(
                self.X_info, X_categories_list_str
            )
            self.X_test_info = process_categories(
                self.X_test_info, X_test_categories_list_str
            )

    def setup_extract_tokens(
        self,
        train_parsed_html_path: Optional[str] = None,
        test_parsed_html_path: Optional[str] = None,
        preprocessor: Optional[preprocess.Preprocessor] = None,
        train_tokens_path: Optional[str] = None,
        test_tokens_path: Optional[str] = None,
        tokenizer: Optional[preprocess.Tokenizer] = None,
    ):
        self.train_parsed_html_path = train_parsed_html_path
        self.test_parsed_html_path = test_parsed_html_path
        self.preprocessor = preprocessor
        self.train_tokens_path = train_tokens_path
        self.test_tokens_path = test_tokens_path
        self.tokenizer = tokenizer
        return self

    def extract_tokens(self):
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not set")
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set")

        if self.train_parsed_html_path is not None:
            self.X_parsed_html = do_or_load(
                self.train_parsed_html_path,
                lambda: self.preprocessor.process(self.X_contents),
            )
        else:
            self.X_parsed_html = self.preprocessor.process(self.X_contents)
        if self.test_parsed_html_path is not None:
            self.X_test_parsed_html = do_or_load(
                self.test_parsed_html_path,
                lambda: self.preprocessor.process(self.X_test_contents),
            )
        else:
            self.X_test_parsed_html = self.preprocessor.process(
                self.X_test_contents
            )
        if self.train_tokens_path is not None:
            self.X_tokens = do_or_load(
                self.train_tokens_path,
                lambda: self.tokenizer.process(self.X_parsed_html),
            )
        else:
            self.X_tokens = self.tokenizer.process(self.X_parsed_html)
        if self.test_tokens_path is not None:
            self.X_test_tokens = do_or_load(
                self.test_tokens_path,
                lambda: self.tokenizer.process(self.X_test_parsed_html),
            )
        else:
            self.X_test_tokens = self.tokenizer.process(
                self.X_test_parsed_html
            )
