"""Encoders and scalers. Fit on training data only; persist so inference reuses them exactly."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """One-hot encode Country (top-N + 'Other') and optionally standardize numeric columns.

    Tree-based models should use scale=False; Logistic Regression, KNN, SVM and neural
    networks should use scale=True.
    """

    def __init__(self, numeric_columns: list[str], top_countries: int = 5) -> None:
        self.numeric_columns = numeric_columns
        self.top_countries = top_countries
        self.countries_: list[str] = []
        self.scaler_ = StandardScaler()

    def fit(self, train: pd.DataFrame) -> FeaturePreprocessor:
        """Learn the country list and scaling statistics from the TRAINING data only."""
        self.countries_ = list(train["country"].value_counts().head(self.top_countries).index)
        self.scaler_.fit(train[self.numeric_columns])
        return self

    def transform(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """Return the model-ready feature matrix."""
        numeric = df[self.numeric_columns].copy()
        if scale:
            numeric = pd.DataFrame(
                self.scaler_.transform(numeric), columns=self.numeric_columns, index=df.index
            )
        country = df["country"].where(df["country"].isin(self.countries_), "Other")
        categories = [*self.countries_, "Other"]
        dummies = pd.get_dummies(pd.Categorical(country, categories=categories), prefix="country")
        dummies.index = df.index
        return pd.concat([numeric, dummies.astype(float)], axis=1)

    def save(self, path: str | Path) -> None:
        """Persist the fitted preprocessor as an artifact."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> FeaturePreprocessor:
        """Load a persisted preprocessor."""
        loaded: FeaturePreprocessor = joblib.load(path)
        return loaded
