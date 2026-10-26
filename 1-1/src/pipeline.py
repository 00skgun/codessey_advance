"""Explicit preprocessing and RFM; only NumPy, pandas and the standard library."""
from pathlib import Path

import numpy as np
import pandas as pd


class DataAnalyzer:
    """Keep raw accounting data separate from imputed / clipped EDA data."""

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.raw = None
        self.df = None
        self.audit = {}
        self.reference_date = None

    def load_data(self, data=None):
        if data is None:
            if self.data_path is None:
                raise ValueError("Provide a CSV path or a DataFrame")
            data = pd.read_csv(Path(self.data_path), dtype={
                "order_id": "string", "product_id": "string", "customer_id": "string"})
        required = {"order_id", "product_id", "product_name", "quantity", "order_date",
                    "price", "customer_id", "country"}
        if not required.issubset(data.columns):
            raise ValueError("Missing columns: " + str(sorted(required - set(data.columns))))
        df = data.copy(deep=True)
        for col in ("order_id", "product_id", "customer_id"):
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
        for col in ("product_name", "country"):
            df[col] = df[col].astype("string")
        for col in ("quantity", "price"):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_numeric_dtype(df.order_date):
            df["order_date"] = pd.to_datetime(df.order_date, unit="D", origin="1899-12-30", errors="coerce")
        else:
            df["order_date"] = pd.to_datetime(df.order_date, errors="coerce", format="mixed")
        self.raw = df.copy(deep=True)
        # Exact duplicates may be legitimate lines; flag them but retain them.
        self.audit = {"raw_rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum())}
        df["price_was_invalid"] = df.price.isna() | df.price.le(0)
        df.loc[df.price_was_invalid, "price"] = np.nan
        df["country"] = df.country.fillna("Unknown")
        self.df = df
        return self.df

    def handle_missing_values(self, strategy="median", group_col="product_id", column="price"):
        """Group transform, then global fallback; identifiers are never imputed."""
        if strategy not in {"median", "mean"}:
            raise ValueError("strategy must be median or mean")
        values = self.df[column]
        if not values.notna().any():
            raise ValueError("Cannot impute a column with no valid observations")
        global_value = getattr(values, strategy)()
        if pd.isna(global_value):
            raise ValueError("Cannot impute a column with no valid observations")
        group_values = self.df.groupby(group_col, dropna=False)[column].transform(strategy)
        self.audit["imputed_" + column] = int(values.isna().sum())
        self.df[column] = values.fillna(group_values).fillna(global_value)
        self.df["product_name"] = self.df.groupby("product_id")["product_name"].transform(
            lambda series: series.ffill().bfill()).fillna("UNKNOWN PRODUCT")
        return self.df

    @staticmethod
    def image_features(images):
        """Vectorized N,H,W,C statistics; no loops over images or pixels."""
        arr = np.asarray(images)
        if arr.ndim != 4 or min(arr.shape) == 0:
            raise ValueError("Expected a nonempty N,H,W,C array")
        if not np.isfinite(arr).all():
            raise ValueError("Images must contain finite numbers")
        arr = arr.astype(np.float32)
        features = pd.DataFrame({
            "image_mean": arr.mean(axis=(1, 2, 3)),
            "image_std": arr.std(axis=(1, 2, 3)),
        })
        # Bonus: simple horizontal/vertical absolute-gradient edge strength.
        if arr.shape[1] > 1 and arr.shape[2] > 1:
            features["image_edge"] = (
                np.abs(np.diff(arr, axis=1)).mean(axis=(1, 2, 3))
                + np.abs(np.diff(arr, axis=2)).mean(axis=(1, 2, 3))) / 2
        else:
            features["image_edge"] = 0.0
        return features

    def engineer_features(self, product_ids, images):
        if len(product_ids) != len(images) or pd.Index(product_ids).has_duplicates:
            raise ValueError("Each unique product must have exactly one image")
        features = self.image_features(images)
        features.index = pd.Index(product_ids, name="product_id")
        self.df = self.df.join(features, on="product_id", validate="many_to_one")
        if self.df.image_mean.isna().any():
            raise ValueError("Image missing for at least one product")
        text = self.df.product_name.fillna("").astype("string")
        self.df["name_length"] = text.str.len()
        self.df["name_word_count"] = text.str.split().str.len()
        self.df["amount"] = self.df.price * self.df.quantity
        return self.df

    def detect_outliers(self, column, threshold=1.5, treatment="flag"):
        """Compute Tukey fences explicitly. Clip a new column, never raw values."""
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        if treatment not in {"flag", "clip"}:
            raise ValueError("treatment must be flag or clip")
        q1, q3 = self.df[column].quantile([0.25, 0.75])
        if pd.isna(q1) or pd.isna(q3):
            raise ValueError("No valid observations")
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        mask = self.df[column].lt(lower) | self.df[column].gt(upper)
        self.df[column + "_outlier"] = mask
        if treatment == "clip":
            self.df[column + "_clipped"] = self.df[column].clip(lower, upper)
        self.audit[column + "_iqr"] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower": float(lower), "upper": float(upper), "count": int(mask.sum())}
        return self.df.loc[mask].copy()

    def valid_transactions(self):
        """Positive, identified purchases; gross spend, not refund-adjusted revenue."""
        df = self.raw
        cancel = df.order_id.str.upper().str.startswith("C", na=False)
        valid = (df.order_id.notna() & df.product_id.notna() & df.customer_id.notna()
                 & df.order_date.notna() & df.price.gt(0) & df.quantity.gt(0) & ~cancel)
        self.audit.update({
            "missing_customer_rows": int(df.customer_id.isna().sum()),
            "cancel_or_nonpositive_qty_rows": int((cancel | df.quantity.le(0)).sum()),
            "nonpositive_or_missing_price_rows": int((df.price.isna() | df.price.le(0)).sum()),
            "invalid_date_rows": int(df.order_date.isna().sum()),
            "valid_purchase_rows": int(valid.sum()), "excluded_rows": int((~valid).sum()),
        })
        result = df.loc[valid].copy()
        result["amount"] = result.price * result.quantity
        return result

    @staticmethod
    def _score(series):
        # Average ranks preserve ties; fewer than five occupied bins is valid.
        return np.ceil(series.rank(method="average", pct=True) * 5).clip(1, 5).astype(int)

    def calculate_rfm(self, customer_col="customer_id", date_col="order_date",
                      amount_col="amount", reference_date=None):
        transactions = self.valid_transactions()
        if transactions.empty:
            raise ValueError("No valid identified purchases for RFM")
        latest = transactions[date_col].max().normalize()
        reference = latest if reference_date is None else pd.Timestamp(reference_date).normalize()
        if pd.isna(reference) or reference < latest:
            raise ValueError("Reference date cannot precede the latest purchase")
        self.reference_date = reference
        rfm = transactions.groupby(customer_col).agg(
            last_purchase=(date_col, "max"), frequency=("order_id", "nunique"),
            monetary=(amount_col, "sum"))
        rfm["recency"] = (reference - rfm.last_purchase.dt.normalize()).dt.days
        rfm["r_score"] = 6 - self._score(rfm.recency)
        rfm["f_score"] = self._score(rfm.frequency)
        rfm["m_score"] = self._score(rfm.monetary)
        rfm["segment"] = np.select([
            rfm.r_score.ge(4) & rfm.f_score.ge(4) & rfm.m_score.ge(4),
            rfm.r_score.le(2),
            rfm.r_score.ge(4) & rfm.frequency.eq(1),
            rfm.f_score.ge(3),
        ], ["VIP", "Churned", "New", "Loyal"], default="Potential")
        return rfm

    def summary_statistics(self, columns=None):
        if columns is None:
            columns = ["price", "quantity", "amount", "name_length", "image_mean", "image_std"]
        return self.df[columns].describe(percentiles=[0.25, 0.5, 0.75]).T
