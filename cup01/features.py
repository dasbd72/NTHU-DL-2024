from typing import List, Optional

import numpy as np
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)

import preprocess
from utils import do_or_load


class Features(object):
    def __init__(
        self,
        X_contents: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_test_contents: Optional[np.ndarray] = None,
        extractor: Optional[preprocess.Extractor] = None,
        train_datetime_path: Optional[str] = None,
        test_datetime_path: Optional[str] = None,
        train_channel_path: Optional[str] = None,
        test_channel_path: Optional[str] = None,
        train_counts_path: Optional[str] = None,
        test_counts_path: Optional[str] = None,
        train_categories_path: Optional[str] = None,
        test_categories_path: Optional[str] = None,
        categories_train_min: int = 1,
        categories_test_min: int = 1,
    ):
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
        # Datetime features
        self.train_datetime_path = train_datetime_path
        self.test_datetime_path = test_datetime_path
        self.datetime_columns: Optional[List[str]] = None
        self.X_datetime_int: Optional[np.ndarray] = None
        self.X_datetime: Optional[np.ndarray] = None  # Final features
        self.X_test_datetime_int: Optional[np.ndarray] = None
        self.X_test_datetime: Optional[np.ndarray] = None  # Final features
        self.sc_datetime = StandardScaler()
        # Channel features
        self.train_channel_path = train_channel_path
        self.test_channel_path = test_channel_path
        self.channel_columns: Optional[List[str]] = None
        self.X_channel_str: Optional[np.ndarray] = None
        self.X_channel_onehot: Optional[np.ndarray] = None
        self.X_channel: Optional[np.ndarray] = None  # Final features
        self.X_test_channel_str: Optional[np.ndarray] = None
        self.X_test_channel_onehot: Optional[np.ndarray] = None
        self.X_test_channel: Optional[np.ndarray] = None  # Final features
        self.onehot_channel = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.sc_channel = StandardScaler()
        # Counts features
        self.train_counts_path = train_counts_path
        self.test_counts_path = test_counts_path
        self.counts_columns: Optional[List[str]] = None
        self.X_counts_int: Optional[np.ndarray] = None
        self.X_counts: Optional[np.ndarray] = None  # Final features
        self.X_test_counts_int: Optional[np.ndarray] = None
        self.X_test_counts: Optional[np.ndarray] = None  # Final features
        self.sc_counts = StandardScaler()
        # Categories features
        self.train_categories_path = train_categories_path
        self.test_categories_path = test_categories_path
        self.categories_train_min = categories_train_min
        self.categories_test_min = categories_test_min
        self.categories_columns: Optional[List[str]] = None
        self.catefories_counts: Optional[np.ndarray] = None
        self.X_categories_list_str: Optional[List] = None
        self.X_categories_mlb: Optional[np.ndarray] = None
        self.X_categories: Optional[np.ndarray] = None  # Final features
        self.X_test_categories_list_str: Optional[List] = None
        self.X_test_categories_mlb: Optional[np.ndarray] = None
        self.X_test_categories: Optional[np.ndarray] = None  # Final features
        self.mlb_categories = MultiLabelBinarizer()
        self.sc_categories = StandardScaler()

    # Extract datetime features
    def extract_datetime_features(self):
        if self.train_datetime_path is not None:
            self.X_datetime_int = do_or_load(
                self.train_datetime_path,
                lambda: self.extractor.extract_datetime(self.X_contents),
            )
        else:
            self.X_datetime_int = self.extractor.extract_datetime(
                self.X_contents
            )
        if self.test_datetime_path is not None:
            self.X_test_datetime_int = do_or_load(
                self.test_datetime_path,
                lambda: self.extractor.extract_datetime(self.X_test_contents),
            )
        else:
            self.X_test_datetime_int = self.extractor.extract_datetime(
                self.X_test_contents
            )
        self.X_datetime = self.sc_datetime.fit_transform(self.X_datetime_int)
        self.X_test_datetime = self.sc_datetime.transform(
            self.X_test_datetime_int
        )
        self.datetime_columns = self.extractor.datetime_columns()

    # Extract channel features
    def extract_channel_features(self):
        if self.train_channel_path is not None:
            self.X_channel_str = do_or_load(
                self.train_channel_path,
                lambda: self.extractor.extract_channel(self.X_contents),
            )
        else:
            self.X_channel_str = self.extractor.extract_channel(
                self.X_contents
            )
        if self.test_channel_path is not None:
            self.X_test_channel_str = do_or_load(
                self.test_channel_path,
                lambda: self.extractor.extract_channel(self.X_test_contents),
            )
        else:
            self.X_test_channel_str = self.extractor.extract_channel(
                self.X_test_contents
            )
        self.X_channel_onehot = self.onehot_channel.fit_transform(
            self.X_channel_str.reshape(-1, 1)
        )
        self.X_test_channel_onehot = self.onehot_channel.transform(
            self.X_test_channel_str.reshape(-1, 1)
        )
        self.X_channel = self.sc_channel.fit_transform(self.X_channel_onehot)
        self.X_test_channel = self.sc_channel.transform(
            self.X_test_channel_onehot
        )
        self.channel_columns = self.onehot_channel.categories_[0]

    # Extract counts features
    def extract_counts_features(self):
        if self.train_counts_path is not None:
            self.X_counts_int = do_or_load(
                self.train_counts_path,
                lambda: self.extractor.extract_counts(self.X_contents),
            )
        else:
            self.X_counts_int = self.extractor.extract_counts(self.X_contents)
        if self.test_counts_path is not None:
            self.X_test_counts_int = do_or_load(
                self.test_counts_path,
                lambda: self.extractor.extract_counts(self.X_test_contents),
            )
        else:
            self.X_test_counts_int = self.extractor.extract_counts(
                self.X_test_contents
            )
        self.X_counts = self.sc_counts.fit_transform(self.X_counts_int)
        self.X_test_counts = self.sc_counts.transform(self.X_test_counts_int)
        self.counts_columns = self.extractor.counts_columns()

    # Extract categories features
    def extract_categories_features(self):
        # Step 1: Extract categories from contents
        if self.train_categories_path is not None:
            self.X_categories_list_str = do_or_load(
                self.train_categories_path,
                lambda: self.extractor.extract_categories(self.X_contents),
            )
        else:
            self.X_categories_list_str = self.extractor.extract_categories(
                self.X_contents
            )
        if self.test_categories_path is not None:
            self.X_test_categories_list_str = do_or_load(
                self.test_categories_path,
                lambda: self.extractor.extract_categories(
                    self.X_test_contents
                ),
            )
        else:
            self.X_test_categories_list_str = (
                self.extractor.extract_categories(self.X_test_contents)
            )

        # Step 2: Count categories
        def count_categories(categories_list_str):
            categories_dict = {}
            for categories in categories_list_str:
                for category in categories:
                    categories_dict[category] = (
                        categories_dict.get(category, 0) + 1
                    )
            return categories_dict

        all_X_categories_dict = count_categories(self.X_categories_list_str)
        all_X_test_categories_dict = count_categories(
            self.X_test_categories_list_str
        )
        # Step 3: Find common categories
        all_X_categories_set = set(all_X_categories_dict.keys())
        all_X_test_categories_set = set(all_X_test_categories_dict.keys())
        all_common_categories_set = all_X_categories_set.intersection(
            all_X_test_categories_set
        )
        # Step 4: Filter common categories with count > categories_min_count
        common_categories_dict = {}
        for category in all_common_categories_set:
            if (
                all_X_categories_dict.get(category, 0)
                >= self.categories_train_min
                and all_X_test_categories_dict.get(category, 0)
                >= self.categories_test_min
            ):
                common_categories_dict[category] = all_X_categories_dict.get(
                    category, 0
                ) + all_X_test_categories_dict.get(category, 0)
        common_categories_set = set(common_categories_dict.keys())
        common_categories = np.array(list(common_categories_set)).astype(str)
        # Step 5: Transform categories to one-hot
        self.mlb_categories.fit(common_categories.reshape(-1, 1))
        self.X_categories_mlb = self.mlb_categories.transform(
            self.X_categories_list_str
        )
        self.X_test_categories_mlb = self.mlb_categories.transform(
            self.X_test_categories_list_str
        )
        # Step 6: Scale categories into final features
        self.X_categories = self.sc_categories.fit_transform(
            self.X_categories_mlb
        )
        self.X_test_categories = self.sc_categories.transform(
            self.X_test_categories_mlb
        )
        self.categories_columns = common_categories
        self.catefories_counts = np.array(
            [
                common_categories_dict[category]
                for category in common_categories
            ]
        )
