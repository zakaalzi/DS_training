import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def _fill_empty_with_nan(df):
        return df.replace({None: np.nan})

    @staticmethod
    def _rename_columns(df):
        rename_dict = dict()
        for col in df.columns:
            if "-" in col:
                rename_dict[col] = col.replace("-", "_")
        rename_dict = {**rename_dict, **{"dellerswebsite": "sellerwebsite"}}
        return df.rename(columns=rename_dict)

    @staticmethod
    def _drop_unused_columns(df):
        cols_to_drop = [
            "model",
            "trim",
            "body_type",
            "generation",
            "manufactured_year",
            "service_history",
            "vrm",
            "co2Emissions",
            "adverttitle",
            "advert",
            "mainimage",
            "images",
            "sellerpostcode",
            "sellerwebsite",
            "year",
            "todaysdate",
            "owners",
            "priceindicators",
            "mileage",
        ]
        return df.drop(columns=cols_to_drop)

    def load_parquet(self, use_nullable=False):
        df = self._rename_columns(
            self._fill_empty_with_nan(
                pd.read_parquet(self.path, use_nullable_dtypes=use_nullable)
            )
        )

        return df

    @staticmethod
    def split_data(df, test_size=0.3, random_seed=1337, as_df=True):
        y = df["price"]
        X = df.drop(columns=["price"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
        if not as_df:
            return (
                np.array(X_train),
                np.array(X_test),
                np.array(y_train),
                np.array(y_test),
            )
        return X_train, X_test, y_train, y_test
