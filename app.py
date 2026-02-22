
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    import plotly.express as px

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
NA_VALUES = ["", "NA", "N/A", "null", "NULL", "-", "?", "None", "none"]
ENCODING_LOW_CARD_MAX_UNIQUE = 15
ENCODING_LOW_CARD_MAX_RATIO = 0.05
KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",
    "AppleGothic",
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]


def configure_korean_matplotlib_font() -> Optional[str]:
    installed_fonts = {f.name for f in fm.fontManager.ttflist}
    for candidate in KOREAN_FONT_CANDIDATES:
        if candidate in installed_fonts:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    return None


class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(arr, 25, axis=0)
        self.q3_ = np.nanpercentile(arr, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * iqr
        self.upper_ = self.q3_ + 1.5 * iqr
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        self.n_features_in_ = df.shape[1]
        self.maps_: List[Dict[Any, float]] = []
        for col_idx in range(df.shape[1]):
            s = pd.Series(df.iloc[:, col_idx]).astype("object")
            freq = s.value_counts(normalize=True, dropna=True).to_dict()
            self.maps_.append(freq)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        if df.shape[1] != getattr(self, "n_features_in_", df.shape[1]):
            raise ValueError("FrequencyEncoder 입력 feature 수가 fit 시점과 다릅니다.")
        encoded_cols = []
        for col_idx in range(df.shape[1]):
            mapped = pd.Series(df.iloc[:, col_idx]).map(self.maps_[col_idx]).fillna(0.0).astype(float)
            encoded_cols.append(mapped.to_numpy())
        if not encoded_cols:
            return np.empty((len(df), 0), dtype=float)
        return np.column_stack(encoded_cols)


def to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    if isinstance(value, tuple):
        return [to_python(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def robust_read_csv(
    uploaded_file, max_rows: int, random_seed: int
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    raw = uploaded_file.getvalue()
    warnings_list: List[str] = []
    errors: List[str] = []
    df = None
    load_meta: Dict[str, Any] = {}

    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                na_values=NA_VALUES,
                keep_default_na=True,
                low_memory=False,
            )
            load_meta = {"encoding": enc, "method": "read_csv(default sep)", "fallback": False}
            break
        except Exception as exc:
            errors.append(f"encoding={enc}: {exc}")

    if df is None:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=None,
                engine="python",
                na_values=NA_VALUES,
                keep_default_na=True,
                low_memory=False,
            )
            load_meta = {
                "encoding": "auto",
                "method": "read_csv(sep=None, engine=python)",
                "fallback": True,
            }
        except Exception as exc:
            errors.append(f"sniffing: {exc}")
            raise ValueError("CSV 로딩 실패\n" + "\n".join(errors)) from exc

    original_rows = len(df)
    if original_rows > max_rows:
        df = df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)
        warnings_list.append(
            f"행 수가 {original_rows:,}개여서 max_rows={max_rows:,} 기준으로 샘플링했습니다."
        )
    else:
        df = df.reset_index(drop=True)

    load_meta["original_rows"] = original_rows
    load_meta["used_rows"] = len(df)
    load_meta["columns"] = len(df.columns)
    return df, load_meta, warnings_list


def robust_read_columns(uploaded_file) -> List[str]:
    raw = uploaded_file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try:
            head = pd.read_csv(io.BytesIO(raw), encoding=enc, nrows=0)
            return head.columns.tolist()
        except Exception:
            continue
    try:
        head = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", nrows=0)
        return head.columns.tolist()
    except Exception:
        return []


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols: List[str] = []
    datetime_cols: List[str] = []
    categorical_cols: List[str] = []
    id_candidates: List[str] = []
    long_text_cols: List[str] = []
    per_column: Dict[str, Any] = {}

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        non_null_count = len(non_null)
        denom = max(non_null_count, 1)
        unique_ratio = float(non_null.nunique(dropna=True) / denom) if non_null_count > 0 else 0.0

        numeric_conv = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() > 0:
            numeric_success = float((numeric_conv.notna() & s.notna()).sum() / s.notna().sum())
        else:
            numeric_success = 0.0
        is_numeric = bool(pd.api.types.is_numeric_dtype(s) or numeric_success >= 0.9)

        datetime_conv = pd.to_datetime(s, errors="coerce")
        if s.notna().sum() > 0:
            datetime_success = float((datetime_conv.notna() & s.notna()).sum() / s.notna().sum())
        else:
            datetime_success = 0.0
        is_datetime = bool((not is_numeric) and (datetime_success >= 0.8))

        is_object_like = bool(
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
            or pd.api.types.is_string_dtype(s)
        )
        is_categorical = bool(is_object_like and unique_ratio < 0.2)
        is_integer_like = bool(pd.api.types.is_integer_dtype(s))
        is_id_candidate = bool((unique_ratio > 0.9) and (is_object_like or is_integer_like) and non_null_count > 0)

        avg_len = float(non_null.astype(str).str.len().mean()) if is_object_like and non_null_count > 0 else 0.0
        is_long_text = bool(is_object_like and avg_len > 50)

        if is_numeric:
            numeric_cols.append(col)
        elif is_datetime:
            datetime_cols.append(col)
        elif is_categorical:
            categorical_cols.append(col)

        if is_id_candidate:
            id_candidates.append(col)
        if is_long_text:
            long_text_cols.append(col)

        inferred_type = "other"
        if is_numeric:
            inferred_type = "numeric"
        elif is_datetime:
            inferred_type = "datetime"
        elif is_categorical:
            inferred_type = "categorical"

        per_column[col] = {
            "raw_dtype": str(s.dtype),
            "inferred_type": inferred_type,
            "numeric_success": round(numeric_success, 4),
            "datetime_success": round(datetime_success, 4),
            "missing_rate": round(float(s.isna().mean()), 4),
            "nunique": int(non_null.nunique(dropna=True)),
            "unique_ratio": round(unique_ratio, 4),
            "avg_str_len": round(avg_len, 2),
            "id_candidate": is_id_candidate,
            "long_text": is_long_text,
        }

    return {
        "numeric_columns": sorted(list(set(numeric_cols))),
        "datetime_columns": sorted(list(set(datetime_cols))),
        "categorical_columns": sorted(list(set(categorical_cols))),
        "id_candidates": sorted(list(set(id_candidates))),
        "long_text_columns": sorted(list(set(long_text_cols))),
        "per_column": per_column,
    }


def numeric_frame(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def compute_outlier_summary(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    num_df = numeric_frame(df, numeric_cols)
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) < 4:
            summary[col] = {"count": 0, "ratio": 0.0, "lower": None, "upper": None}
            continue
        q1 = float(np.percentile(s, 25))
        q3 = float(np.percentile(s, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        summary[col] = {
            "count": int(mask.sum()),
            "ratio": round(float(mask.mean()), 4),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
        }
    return summary


def compute_correlation(num_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if num_df.shape[1] < 2:
        return pd.DataFrame(), []
    corr = num_df.corr(numeric_only=True)
    abs_corr = corr.abs()
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(abs_corr.iloc[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = [
        {"col_a": a, "col_b": b, "abs_corr": round(v, 4), "corr": round(float(corr.loc[a, b]), 4)}
        for a, b, v in pairs[:20]
    ]
    return corr, top_pairs


def detect_task_type(y_raw: pd.Series) -> str:
    y_non_null = y_raw.dropna()
    if len(y_non_null) == 0:
        return "classification"
    y_num = pd.to_numeric(y_non_null, errors="coerce")
    numeric_success = float(y_num.notna().mean())
    unique_count = int(y_non_null.nunique(dropna=True))
    threshold = max(15, int(0.05 * len(y_non_null)))
    if numeric_success >= 0.95 and unique_count > threshold:
        return "regression"
    return "classification"


def build_model_candidates(task_type: str, random_seed: int, n_classes: int, fast_mode: bool):
    n_estimators = 200 if fast_mode else 350
    candidates = []

    if task_type == "classification":
        candidates.append(
            (
                "RandomForest",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_seed,
                    n_jobs=-1,
                    class_weight="balanced" if n_classes > 2 else None,
                ),
                True,
                "",
            )
        )
    else:
        candidates.append(
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
                True,
                "",
            )
        )

    if HAS_XGBOOST:
        if task_type == "classification":
            params = {
                "n_estimators": n_estimators,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_seed,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }
            if n_classes <= 2:
                params["objective"] = "binary:logistic"
            else:
                params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
            candidates.append(("XGBoost", XGBClassifier(**params), True, ""))
        else:
            candidates.append(
                (
                    "XGBoost",
                    XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_seed,
                        objective="reg:squarederror",
                        n_jobs=-1,
                    ),
                    True,
                    "",
                )
            )
    else:
        candidates.append(("XGBoost", None, False, "xgboost 미설치"))

    if HAS_LIGHTGBM:
        if task_type == "classification":
            candidates.append(
                (
                    "LightGBM",
                    LGBMClassifier(
                        n_estimators=n_estimators,
                        random_state=random_seed,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    True,
                    "",
                )
            )
        else:
            candidates.append(
                (
                    "LightGBM",
                    LGBMRegressor(
                        n_estimators=n_estimators,
                        random_state=random_seed,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    True,
                    "",
                )
            )
    else:
        candidates.append(("LightGBM", None, False, "lightgbm 미설치"))

    return candidates


def run_supervised(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    target_col: str,
    test_size: float,
    random_seed: int,
    outlier_option: str,
    model_speed: str,
    warnings_list: List[str],
) -> Dict[str, Any]:
    preprocessing_log: List[Dict[str, Any]] = []

    if target_col not in df.columns:
        raise ValueError("선택한 타겟 컬럼이 데이터에 없습니다.")

    task_type = detect_task_type(df[target_col])
    excluded = set(schema["id_candidates"] + schema["long_text_columns"])
    feature_candidates = [c for c in df.columns if c != target_col and c not in excluded]

    datetime_excluded = [c for c in schema["datetime_columns"] if c in feature_candidates]
    for c in datetime_excluded:
        preprocessing_log.append(
            {"column": c, "stage": "feature_selection", "action": "excluded", "detail": "datetime 컬럼(MVP)"}
        )

    feature_candidates = [c for c in feature_candidates if c not in datetime_excluded]
    numeric_features = [c for c in schema["numeric_columns"] if c in feature_candidates]
    categorical_features = [c for c in feature_candidates if c not in numeric_features]

    for c in schema["id_candidates"]:
        if c != target_col:
            preprocessing_log.append(
                {"column": c, "stage": "feature_selection", "action": "excluded", "detail": "id_candidate"}
            )
    for c in schema["long_text_columns"]:
        if c != target_col:
            preprocessing_log.append(
                {"column": c, "stage": "feature_selection", "action": "excluded", "detail": "long_text"}
            )

    for c in numeric_features:
        detail = "median imputation"
        if outlier_option == "IQR 클리핑":
            detail += " + IQR clipping"
        preprocessing_log.append(
            {"column": c, "stage": "preprocessing", "action": "numeric_pipeline", "detail": detail}
        )

    if len(feature_candidates) == 0:
        raise ValueError("모델 입력으로 사용할 컬럼이 없습니다. (ID/long_text/datetime 제외 후 0개)")

    X = df[feature_candidates].copy()
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_features:
        # pandas NA가 sklearn 내부 비교 연산으로 들어가면 "boolean value of NA is ambiguous"
        # 오류를 유발할 수 있어 object + np.nan으로 정규화한다.
        X[c] = X[c].astype("object")
        X[c] = X[c].where(pd.notna(X[c]), np.nan)

    low_card_cat_features: List[str] = []
    freq_cat_features: List[str] = []
    for c in categorical_features:
        nunique = int(X[c].nunique(dropna=True))
        ratio = float(nunique / max(len(X), 1))
        if nunique <= ENCODING_LOW_CARD_MAX_UNIQUE and ratio <= ENCODING_LOW_CARD_MAX_RATIO:
            low_card_cat_features.append(c)
            preprocessing_log.append(
                {
                    "column": c,
                    "stage": "preprocessing",
                    "action": "categorical_encoding",
                    "detail": f"onehot (nunique={nunique}, ratio={ratio:.4f})",
                }
            )
        else:
            freq_cat_features.append(c)
            preprocessing_log.append(
                {
                    "column": c,
                    "stage": "preprocessing",
                    "action": "categorical_encoding",
                    "detail": f"frequency_encoding (nunique={nunique}, ratio={ratio:.4f})",
                }
            )

    y_raw = df[target_col].copy()
    if task_type == "regression":
        y = pd.to_numeric(y_raw, errors="coerce")
        valid_mask = y.notna()
    else:
        y = y_raw.astype("object")
        valid_mask = y.notna()

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(X) < 30:
        warnings_list.append(f"유효 학습 샘플 수가 적습니다: {len(X)}")

    n_classes = 0
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = y.astype(str)
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)
        n_classes = int(pd.Series(y).nunique())
        if n_classes < 2:
            raise ValueError("분류 타겟 클래스가 2개 미만입니다.")

    stratify_data = None
    if task_type == "classification":
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() >= 2:
            stratify_data = y
        else:
            warnings_list.append("클래스 빈도가 너무 낮아 stratify를 사용하지 않았습니다.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_data if task_type == "classification" else None,
        )
    except ValueError:
        warnings_list.append("stratify 분할에 실패하여 일반 분할로 진행했습니다.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=None,
        )

    num_steps = []
    if outlier_option == "IQR 클리핑":
        num_steps.append(("iqr", IQRClipper()))
    num_steps.append(("imputer", SimpleImputer(strategy="median")))

    cat_low_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    cat_freq_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", missing_values=np.nan)),
            ("freq", FrequencyEncoder()),
        ]
    )

    transformers = [("num", Pipeline(steps=num_steps), numeric_features)]
    if low_card_cat_features:
        transformers.append(("cat_low", cat_low_pipeline, low_card_cat_features))
    if freq_cat_features:
        transformers.append(("cat_freq", cat_freq_pipeline, freq_cat_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    fast_mode = model_speed == "빠름"
    candidates = build_model_candidates(task_type, random_seed, n_classes, fast_mode)

    rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    notes.append(
        f"categorical encoding: onehot={len(low_card_cat_features)}개, frequency={len(freq_cat_features)}개"
    )

    for model_name, estimator, available, reason in candidates:
        if not available:
            rows.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "reason": reason,
                    "accuracy": None,
                    "f1_macro": None,
                    "roc_auc": None,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                }
            )
            notes.append(f"{model_name}: {reason}")
            continue

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        row: Dict[str, Any] = {
            "model": model_name,
            "status": "trained",
            "reason": "",
            "accuracy": None,
            "f1_macro": None,
            "roc_auc": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "cv_metric": None,
            "cv_mean": None,
            "cv_std": None,
        }

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if task_type == "classification":
                row["accuracy"] = float(accuracy_score(y_test, y_pred))
                row["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

                if hasattr(pipeline, "predict_proba"):
                    try:
                        probs = pipeline.predict_proba(X_test)
                        if probs.shape[1] == 2:
                            row["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
                        elif probs.shape[1] > 2:
                            row["roc_auc"] = float(
                                roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
                            )
                    except Exception:
                        row["roc_auc"] = None
                        notes.append(f"{model_name}: roc_auc 계산을 건너뜀")
                else:
                    notes.append(f"{model_name}: predict_proba 미지원으로 roc_auc 건너뜀")

                if not fast_mode:
                    cv = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=5,
                        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                        n_jobs=-1,
                    )
                    row["cv_metric"] = "f1_macro"
                    row["cv_mean"] = float(np.mean(cv["test_f1_macro"]))
                    row["cv_std"] = float(np.std(cv["test_f1_macro"]))

            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                row["rmse"] = rmse
                row["mae"] = mae
                row["r2"] = r2

                if not fast_mode:
                    cv = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=5,
                        scoring={
                            "rmse": "neg_root_mean_squared_error",
                            "mae": "neg_mean_absolute_error",
                            "r2": "r2",
                        },
                        n_jobs=-1,
                    )
                    row["cv_metric"] = "rmse"
                    row["cv_mean"] = float(-np.mean(cv["test_rmse"]))
                    row["cv_std"] = float(np.std(cv["test_rmse"]))

        except Exception as exc:
            row["status"] = "failed"
            row["reason"] = str(exc)
            notes.append(f"{model_name} 실패: {exc}")

        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    return {
        "task": task_type,
        "feature_count": len(feature_candidates),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics_df": metrics_df,
        "metrics_table": to_python(metrics_df.to_dict(orient="records")),
        "notes": notes,
        "preprocessing_log": preprocessing_log,
    }


def run_unsupervised(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    random_seed: int,
    warnings_list: List[str],
) -> Dict[str, Any]:
    preprocessing_log: List[Dict[str, Any]] = []

    excluded = set(schema["id_candidates"] + schema["long_text_columns"])
    numeric_cols = [c for c in schema["numeric_columns"] if c not in excluded]

    for c in schema["id_candidates"]:
        preprocessing_log.append(
            {
                "column": c,
                "stage": "feature_selection",
                "action": "excluded",
                "detail": "id_candidate",
                "why": "unique ratio 높아 학습에 정보보다 식별자 역할 가능",
            }
        )
    for c in schema["long_text_columns"]:
        preprocessing_log.append(
            {
                "column": c,
                "stage": "feature_selection",
                "action": "excluded",
                "detail": "long_text",
                "why": "MVP에서는 텍스트 벡터화 미지원",
            }
        )

    if len(numeric_cols) == 0:
        raise ValueError("비지도 학습에 사용할 numeric 컬럼이 없습니다.")

    num_df = numeric_frame(df, numeric_cols)
    valid_cols = [c for c in num_df.columns if num_df[c].notna().sum() > 0]
    num_df = num_df[valid_cols]

    if num_df.shape[1] == 0:
        raise ValueError("모든 numeric 컬럼이 결측만 포함하고 있습니다.")

    for c in valid_cols:
        preprocessing_log.append(
            {
                "column": c,
                "stage": "preprocessing",
                "action": "numeric_pipeline",
                "detail": "median imputation + standard scaling + pca2d",
                "why": "one-hot 폭발 방지",
            }
        )

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(num_df)

    if X_imp.shape[0] < 3:
        raise ValueError("샘플 수가 3개 미만이라 클러스터링이 불가능합니다.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    max_k = min(8, X_scaled.shape[0] - 1)
    if max_k < 2:
        raise ValueError("유효 샘플 수가 부족해 KMeans를 수행할 수 없습니다.")

    k_scores: List[Dict[str, Any]] = []
    k_eval_errors: List[str] = []
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(np.unique(labels)) < 2:
                continue
            sil = float(silhouette_score(X_scaled, labels))
            dbi = float(davies_bouldin_score(X_scaled, labels))
            ch = float(calinski_harabasz_score(X_scaled, labels))
            k_scores.append(
                {
                    "k": k,
                    "silhouette": sil,
                    "davies_bouldin": dbi,
                    "calinski_harabasz": ch,
                    "inertia": float(km.inertia_),
                    "labels": labels,
                }
            )
        except Exception as exc:
            k_eval_errors.append(f"k={k} 평가 실패: {exc}")
            continue

    if not k_scores:
        raise ValueError("유효한 클러스터 분할을 찾지 못했습니다.")

    k_df = pd.DataFrame(k_scores)
    safe_df = k_df.copy()
    metric_cols = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    for col in metric_cols:
        safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce")
        safe_df[col] = safe_df[col].replace([np.inf, -np.inf], np.nan)

    rank_df = safe_df.dropna(subset=metric_cols).copy()
    if rank_df.empty:
        fallback_idx = safe_df["silhouette"].fillna(-np.inf).idxmax()
        chosen_k = int(safe_df.loc[fallback_idx, "k"])
        choice_reason = "지표 일부가 유효하지 않아 silhouette 최대값 기준 fallback 선택을 사용했습니다."
        safe_df["rank_sil"] = np.nan
        safe_df["rank_db"] = np.nan
        safe_df["rank_ch"] = np.nan
        safe_df["total_score"] = np.nan
    else:
        rank_df["rank_sil"] = rank_df["silhouette"].rank(method="min", ascending=False)
        rank_df["rank_db"] = rank_df["davies_bouldin"].rank(method="min", ascending=True)
        rank_df["rank_ch"] = rank_df["calinski_harabasz"].rank(method="min", ascending=False)
        w1, w2, w3 = 0.45, 0.35, 0.20
        rank_df["total_score"] = w1 * rank_df["rank_sil"] + w2 * rank_df["rank_db"] + w3 * rank_df["rank_ch"]
        rank_df = rank_df.sort_values(["total_score", "rank_sil", "k"], ascending=[True, True, True]).reset_index(drop=True)
        chosen_k = int(rank_df.iloc[0]["k"])
        safe_df = safe_df.merge(
            rank_df[["k", "rank_sil", "rank_db", "rank_ch", "total_score"]],
            on="k",
            how="left",
        )
        choice_reason = ""

    sil_series = safe_df["silhouette"].dropna()
    db_series = safe_df["davies_bouldin"].dropna()
    ch_series = safe_df["calinski_harabasz"].dropna()
    best_sil_k = int(safe_df.loc[sil_series.idxmax(), "k"]) if not sil_series.empty else chosen_k
    best_db_k = int(safe_df.loc[db_series.idxmin(), "k"]) if not db_series.empty else chosen_k
    best_ch_k = int(safe_df.loc[ch_series.idxmax(), "k"]) if not ch_series.empty else chosen_k
    best_by_metric = {
        "silhouette": best_sil_k,
        "davies_bouldin": best_db_k,
        "calinski_harabasz": best_ch_k,
    }
    consensus = len(set(best_by_metric.values())) == 1
    if not choice_reason:
        if consensus:
            choice_reason = (
                f"지표 합의(consensus=true): silhouette/DB/CH 모두 k={chosen_k}를 지지하여 선택했습니다."
            )
        else:
            choice_reason = (
                f"지표 합의 없음(consensus=false). 가중 랭크 합산(0.45/0.35/0.20) 기준 최소 점수인 k={chosen_k}를 선택했습니다."
            )

    chosen_metric_row = safe_df.loc[safe_df["k"] == chosen_k].iloc[0]
    chosen_silhouette = float(chosen_metric_row["silhouette"]) if pd.notna(chosen_metric_row["silhouette"]) else -1.0
    if chosen_silhouette < 0.2:
        warnings_list.append(
            f"군집 구조 약함: chosen_k={chosen_k}의 silhouette={chosen_silhouette:.4f} (<0.2)"
        )

    labels_selected = None
    for row in k_scores:
        if int(row["k"]) == chosen_k:
            labels_selected = row["labels"]
            break
    if labels_selected is None:
        fallback_row = max(k_scores, key=lambda x: x.get("silhouette", -np.inf))
        chosen_k = int(fallback_row["k"])
        labels_selected = fallback_row["labels"]
        warnings_list.append("chosen_k labels 매핑 실패로 silhouette 기준 fallback labels를 사용했습니다.")
        chosen_metric_row = safe_df.loc[safe_df["k"] == chosen_k].iloc[0]

    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "cluster": labels_selected,
        }
    )

    work_df = num_df.copy().reset_index(drop=True)
    work_df["cluster"] = labels_selected

    cluster_sizes = (
        work_df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="size")
    )
    cluster_mean = work_df.groupby("cluster")[valid_cols].mean(numeric_only=True).reset_index()
    cluster_median = work_df.groupby("cluster")[valid_cols].median(numeric_only=True).reset_index()
    cluster_mean = cluster_mean.sort_values("cluster").reset_index(drop=True)
    cluster_median = cluster_median.sort_values("cluster").reset_index(drop=True)
    cluster_ids = cluster_mean["cluster"].astype(int).tolist()

    feature_gaps: List[Dict[str, Any]] = []
    if len(cluster_ids) == 2:
        c0, c1 = cluster_ids[0], cluster_ids[1]
        m0 = cluster_mean[cluster_mean["cluster"] == c0].iloc[0]
        m1 = cluster_mean[cluster_mean["cluster"] == c1].iloc[0]
        gap_rows = []
        for feat in valid_cols:
            mean_c0 = float(m0[feat])
            mean_c1 = float(m1[feat])
            gap = abs(mean_c0 - mean_c1)
            direction = "c0>c1" if mean_c0 > mean_c1 else "c1>c0"
            gap_rows.append(
                {
                    "feature": feat,
                    "mean_c0": mean_c0,
                    "mean_c1": mean_c1,
                    "gap": gap,
                    "direction": direction,
                }
            )
        feature_gaps = sorted(gap_rows, key=lambda x: x["gap"], reverse=True)[:5]
    else:
        spread_rows = []
        for feat in valid_cols:
            values = cluster_mean[["cluster", feat]].copy()
            max_idx = values[feat].idxmax()
            min_idx = values[feat].idxmin()
            max_cluster = int(values.loc[max_idx, "cluster"])
            min_cluster = int(values.loc[min_idx, "cluster"])
            max_val = float(values.loc[max_idx, feat])
            min_val = float(values.loc[min_idx, feat])
            spread_rows.append(
                {
                    "feature": feat,
                    "spread": max_val - min_val,
                    "max_cluster": max_cluster,
                    "min_cluster": min_cluster,
                    "values_by_cluster": {
                        str(int(r["cluster"])): float(r[feat]) for _, r in values.iterrows()
                    },
                }
            )
        feature_gaps = sorted(spread_rows, key=lambda x: x["spread"], reverse=True)[:5]

    metrics_table = safe_df[
        [
            "k",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "inertia",
            "rank_sil",
            "rank_db",
            "rank_ch",
            "total_score",
        ]
    ].copy()
    metrics_table = metrics_table.sort_values("k").reset_index(drop=True)

    k_selection = {
        "k_candidates": [int(k) for k in metrics_table["k"].tolist()],
        "metrics_table": to_python(metrics_table.to_dict(orient="records")),
        "best_by_metric": best_by_metric,
        "consensus": bool(consensus),
        "chosen_k": chosen_k,
        "choice_reason": choice_reason,
    }

    notes = [
        f"best_by_metric={best_by_metric}",
        f"consensus={consensus}",
        f"chosen_k={chosen_k}",
        f"choice_reason={choice_reason}",
    ]
    notes.extend(k_eval_errors)
    return {
        "metrics": {
            "k": chosen_k,
            "silhouette": float(chosen_metric_row["silhouette"]) if pd.notna(chosen_metric_row["silhouette"]) else None,
            "davies_bouldin": float(chosen_metric_row["davies_bouldin"]) if pd.notna(chosen_metric_row["davies_bouldin"]) else None,
            "calinski_harabasz": float(chosen_metric_row["calinski_harabasz"]) if pd.notna(chosen_metric_row["calinski_harabasz"]) else None,
        },
        "k_scores": to_python(metrics_table.to_dict(orient="records")),
        "cluster_sizes_df": cluster_sizes,
        "cluster_sizes": to_python(cluster_sizes.to_dict(orient="records")),
        "cluster_mean_df": cluster_mean,
        "cluster_median_df": cluster_median,
        "cluster_profile_mean": to_python(cluster_mean.to_dict(orient="records")),
        "cluster_profile_median": to_python(cluster_median.to_dict(orient="records")),
        "cluster_feature_gaps": to_python(feature_gaps),
        "k_selection": to_python(k_selection),
        "k_curve_df": metrics_table[["k", "silhouette", "inertia"]].copy(),
        "pca_df": pca_df,
        "numeric_columns": valid_cols,
        "notes": notes,
        "preprocessing_log": preprocessing_log,
    }


def build_report_data(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    missing_df: pd.DataFrame,
    outlier_summary: Dict[str, Any],
    corr_pairs: List[Dict[str, Any]],
    preprocessing_log: List[Dict[str, Any]],
    warnings_list: List[str],
    supervised_result: Optional[Dict[str, Any]],
    unsupervised_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    supervised_block = {
        "task": "N/A",
        "metrics_table": [],
        "notes": [],
    }
    if supervised_result is not None:
        supervised_block = {
            "task": supervised_result["task"],
            "metrics_table": supervised_result["metrics_table"],
            "notes": supervised_result["notes"],
        }

    unsupervised_block = {
        "metrics": {},
        "cluster_sizes": [],
        "notes": [],
        "k_selection": {},
        "cluster_feature_gaps": [],
    }
    if unsupervised_result is not None:
        unsupervised_block = {
            "metrics": unsupervised_result["metrics"],
            "cluster_sizes": unsupervised_result["cluster_sizes"],
            "notes": unsupervised_result["notes"],
            "k_selection": unsupervised_result.get("k_selection", {}),
            "cluster_feature_gaps": unsupervised_result.get("cluster_feature_gaps", []),
        }

    report_data = {
        "overview": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "dtypes": to_python(df.dtypes.astype(str).to_dict()),
        },
        "schema": schema,
        "missing": to_python(missing_df.to_dict(orient="records")),
        "outliers": outlier_summary,
        "correlation": {"top_pairs": corr_pairs},
        "preprocessing_log": preprocessing_log,
        "supervised": supervised_block,
        "unsupervised": unsupervised_block,
        "cluster_profile_mean": (
            unsupervised_result.get("cluster_profile_mean", []) if unsupervised_result is not None else []
        ),
        "cluster_profile_median": (
            unsupervised_result.get("cluster_profile_median", []) if unsupervised_result is not None else []
        ),
        "cluster_feature_gaps": (
            unsupervised_result.get("cluster_feature_gaps", []) if unsupervised_result is not None else []
        ),
        "k_selection": unsupervised_result.get("k_selection", {}) if unsupervised_result is not None else {},
        "warnings": warnings_list,
    }
    return to_python(report_data)


def generate_llm_report(report_data: Dict[str, Any], head_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY가 없어 LLM 리포트 기능이 비활성화되었습니다."

    try:
        from openai import OpenAI
    except Exception as exc:
        return None, f"openai 패키지 로딩 실패: {exc}"

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "너는 데이터 분석 리포트 작성자다. "
        "숫자는 report_data JSON에 있는 값만 인용하라. 새로운 수치, 통계, 비율을 절대 만들어내지 마라. "
        "추상적 문장만 쓰지 말고 반드시 수치 인용을 포함하라. "
        "근거가 불충분하면 '추정' 또는 '불확실'이라고 명시하라. "
        "출력은 한국어 Markdown으로 작성한다."
    )

    user_prompt = f"""
아래 데이터를 기반으로 Markdown 리포트를 작성하라.

강제 규칙:
- 숫자는 report_data JSON에 있는 값만 인용하라.
- 새로운 수치/통계를 만들어내지 마라.
- 추상적 문장만 쓰지 말고 반드시 수치 인용을 포함하라.
- 추정 또는 불확실한 내용은 명시하라.

필수 섹션:
1) 도메인 추정(추정임을 명시)
2) 주요 컬럼 설명(추정, 불확실성 명시)
3) 결측 처리 요약 + 대안 1~2개
4) 이상치 처리 요약 + 대안
5) EDA 인사이트(상관/분포/편향)
6) 모델 결과 해석(지도/비지도)
7) 다음 실험 추천
8) 리스크/주의(누수, 표본수, 불균형, 편향)

비지도(클러스터링) 관련 필수 작성 규칙:
- k_selection을 반드시 사용해 chosen_k 선택 근거를 설명하라.
  : consensus(true/false), best_by_metric, chosen_k, choice_reason를 모두 언급하라.
- cluster_feature_gaps Top 5를 반드시 사용해 클러스터 차이를 수치로 비교하는 문단을 작성하라.
  : 예시 형식처럼 cluster별 평균값과 gap/spread를 숫자로 인용하라.
- 아래 문장을 반드시 포함하라.
  : "상관은 인과가 아니다"
  : "time_on_app 등 파생 변수 가능성"

[report_data JSON]
{json.dumps(report_data, ensure_ascii=False, indent=2)}

[df.head(10)]
{head_df.to_string(index=False)}
"""

    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ],
        )
        text = response.output_text
        if text:
            return text, None
    except Exception:
        pass

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        if isinstance(text, list):
            text = "\n".join([str(x) for x in text])
        return text, None
    except Exception as exc:
        return None, f"LLM 호출 실패: {exc}"


def run_analysis(
    uploaded_file,
    mode: str,
    target_col: Optional[str],
    max_rows: int,
    test_size: float,
    random_seed: int,
    outlier_option: str,
    model_speed: str,
) -> Dict[str, Any]:
    warnings_list: List[str] = []
    df, load_meta, load_warnings = robust_read_csv(uploaded_file, max_rows=max_rows, random_seed=random_seed)
    warnings_list.extend(load_warnings)

    schema = infer_schema(df)
    num_df = numeric_frame(df, schema["numeric_columns"])

    missing_df = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_rate": (df.isna().mean() * 100).round(2).values,
            }
        )
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )

    outlier_summary = compute_outlier_summary(df, schema["numeric_columns"])
    corr_matrix, corr_pairs = compute_correlation(num_df)

    supervised_result = None
    unsupervised_result = None
    preprocessing_log: List[Dict[str, Any]] = []

    if mode == "지도학습":
        supervised_result = run_supervised(
            df=df,
            schema=schema,
            target_col=target_col or "",
            test_size=test_size,
            random_seed=random_seed,
            outlier_option=outlier_option,
            model_speed=model_speed,
            warnings_list=warnings_list,
        )
        preprocessing_log.extend(supervised_result["preprocessing_log"])
    else:
        unsupervised_result = run_unsupervised(
            df=df,
            schema=schema,
            random_seed=random_seed,
            warnings_list=warnings_list,
        )
        preprocessing_log.extend(unsupervised_result["preprocessing_log"])

    report_data = build_report_data(
        df=df,
        schema=schema,
        missing_df=missing_df,
        outlier_summary=outlier_summary,
        corr_pairs=corr_pairs,
        preprocessing_log=preprocessing_log,
        warnings_list=warnings_list,
        supervised_result=supervised_result,
        unsupervised_result=unsupervised_result,
    )

    return {
        "df": df,
        "schema": schema,
        "num_df": num_df,
        "missing_df": missing_df,
        "outlier_summary": outlier_summary,
        "corr_matrix": corr_matrix,
        "corr_pairs": corr_pairs,
        "mode": mode,
        "supervised": supervised_result,
        "unsupervised": unsupervised_result,
        "preprocessing_log": preprocessing_log,
        "warnings": warnings_list,
        "load_meta": load_meta,
        "report_data": report_data,
    }


def draw_overview(result: Dict[str, Any]):
    df = result["df"]
    schema = result["schema"]
    load_meta = result["load_meta"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric Columns", f"{len(schema['numeric_columns'])}")

    st.write("**CSV 로딩 정보**")
    st.json(load_meta)

    dtype_summary = (
        pd.Series({k: v["raw_dtype"] for k, v in schema["per_column"].items()})
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="count")
    )
    st.write("**컬럼 타입 요약(raw dtype 기준)**")
    st.dataframe(dtype_summary, use_container_width=True)

    st.write("**상위 20행**")
    st.dataframe(df.head(20), use_container_width=True)


def draw_eda(result: Dict[str, Any]):
    df = result["df"]
    schema = result["schema"]
    missing_df = result["missing_df"]

    st.write("**결측률**")
    st.dataframe(missing_df, use_container_width=True)

    st.write("**수치형 분포 (Histogram + Boxplot)**")
    if len(schema["numeric_columns"]) > 0:
        selected_num = st.selectbox("수치형 컬럼 선택", schema["numeric_columns"], key="eda_numeric_col")
        col_data = pd.to_numeric(df[selected_num], errors="coerce")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        sns.histplot(col_data.dropna(), bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram: {selected_num}")
        sns.boxplot(x=col_data.dropna(), ax=axes[1])
        axes[1].set_title(f"Boxplot: {selected_num}")
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("수치형 컬럼이 없습니다.")

    st.write("**범주형 Top 빈도**")
    cat_candidates = schema["categorical_columns"]
    if len(cat_candidates) > 0:
        selected_cat = st.selectbox("범주형 컬럼 선택", cat_candidates, key="eda_cat_col")
        freq_df = (
            df[selected_cat]
            .astype("string")
            .fillna("<NA>")
            .value_counts(dropna=False)
            .head(20)
            .rename_axis(selected_cat)
            .reset_index(name="count")
        )
        st.dataframe(freq_df, use_container_width=True)
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        sns.barplot(data=freq_df, x=selected_cat, y="count", ax=ax2)
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_title(f"Top Categories: {selected_cat}")
        st.pyplot(fig2, use_container_width=False)
    else:
        st.info("규칙 기반 categorical 컬럼이 없습니다.")

    st.write("**수치형 컬럼 상관 히트맵**")
    if len(schema["numeric_columns"]) >= 2:
        default_corr_cols = schema["numeric_columns"][: min(10, len(schema["numeric_columns"]))]
        corr_cols = st.multiselect(
            "상관 히트맵 컬럼 선택",
            options=schema["numeric_columns"],
            default=default_corr_cols,
            key="eda_corr_cols",
        )
        if len(corr_cols) >= 2:
            corr_df = df[corr_cols].apply(pd.to_numeric, errors="coerce")
            corr_matrix = corr_df.corr(numeric_only=True)
            # 대각선(자기상관=1)과 상삼각을 숨겨 실제 컬럼간 상관만 강조
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
            fig3, ax3 = plt.subplots(figsize=(max(5.5, len(corr_cols) * 0.55), max(4.0, len(corr_cols) * 0.45)))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                ax=ax3,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8},
                linewidths=0.4,
                square=True,
                cbar_kws={"shrink": 0.8},
            )
            ax3.set_title("Correlation Heatmap (Lower Triangle)")
            st.pyplot(fig3, use_container_width=False)
            abs_pairs = []
            cols = corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    abs_pairs.append(
                        {
                            "col_a": cols[i],
                            "col_b": cols[j],
                            "corr": float(corr_matrix.iloc[i, j]),
                            "abs_corr": abs(float(corr_matrix.iloc[i, j])),
                        }
                    )
            pair_df = pd.DataFrame(abs_pairs).sort_values("abs_corr", ascending=False).head(10)
            if not pair_df.empty:
                pair_df["corr"] = pair_df["corr"].round(4)
                pair_df["abs_corr"] = pair_df["abs_corr"].round(4)
                st.write("상위 상관쌍 (|corr| 기준)")
                st.dataframe(pair_df, use_container_width=True)
        else:
            st.info("상관 히트맵은 수치형 컬럼 2개 이상 선택해야 합니다.")
    else:
        st.info("상관 히트맵을 그릴 수 있는 수치형 컬럼이 2개 미만입니다.")


def draw_preprocessing_log(result: Dict[str, Any]):
    logs = result["preprocessing_log"]
    if not logs:
        st.info("전처리 로그가 없습니다.")
        return
    log_df = pd.DataFrame(logs)
    if "why" not in log_df.columns:
        log_df["why"] = ""
    st.dataframe(log_df, use_container_width=True)


def draw_modeling(result: Dict[str, Any]):
    mode = result["mode"]

    if mode == "지도학습":
        sup = result["supervised"]
        if sup is None:
            st.info("지도학습 결과가 없습니다.")
            return

        st.write(f"**Task Type:** `{sup['task']}`")
        st.write(f"Train/Test: {sup['train_size']} / {sup['test_size']} | Feature count: {sup['feature_count']}")

        metrics_df = sup["metrics_df"].copy()
        st.dataframe(metrics_df, use_container_width=True)

        trained_df = metrics_df[metrics_df["status"] == "trained"].copy()
        if not trained_df.empty:
            metric_col = "f1_macro" if sup["task"] == "classification" else "r2"
            plot_df = trained_df[["model", metric_col]].dropna()
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(data=plot_df, x="model", y=metric_col, ax=ax)
                ax.set_title(f"Model Comparison ({metric_col})")
                st.pyplot(fig, use_container_width=False)

        if sup["notes"]:
            st.write("**Notes**")
            for note in sup["notes"]:
                st.write(f"- {note}")

    else:
        unsup = result["unsupervised"]
        if unsup is None:
            st.info("비지도학습 결과가 없습니다.")
            return

        st.write("**PCA 2D Scatter**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=unsup["pca_df"], x="PC1", y="PC2", hue="cluster", palette="tab10", ax=ax)
        ax.set_title("PCA 2D by Cluster")
        st.pyplot(fig, use_container_width=False)

        st.write("**클러스터링 지표**")
        st.json(unsup["metrics"])

        st.write("**K 탐색 결과**")
        st.dataframe(pd.DataFrame(unsup["k_scores"]), use_container_width=True)

        k_sel = unsup.get("k_selection", {})
        if k_sel:
            st.write("**K Selection**")
            st.json(k_sel)

        k_curve_df = unsup.get("k_curve_df")
        if isinstance(k_curve_df, pd.DataFrame) and not k_curve_df.empty:
            st.write("**Elbow (K vs Inertia)**")
            if HAS_PLOTLY:
                fig_elbow = px.line(
                    k_curve_df,
                    x="k",
                    y="inertia",
                    markers=True,
                    title="Elbow Curve",
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
            else:
                st.line_chart(k_curve_df.set_index("k")[["inertia"]], use_container_width=True)

            st.write("**K vs Silhouette**")
            if HAS_PLOTLY:
                fig_sil = px.line(
                    k_curve_df,
                    x="k",
                    y="silhouette",
                    markers=True,
                    title="Silhouette by K",
                )
                st.plotly_chart(fig_sil, use_container_width=True)
            else:
                st.line_chart(k_curve_df.set_index("k")[["silhouette"]], use_container_width=True)

        st.write("**Cluster Sizes**")
        st.dataframe(unsup["cluster_sizes_df"], use_container_width=True)

        st.write("**Cluster Profile (Mean)**")
        st.dataframe(unsup["cluster_mean_df"], use_container_width=True)

        st.write("**Cluster Profile (Median)**")
        st.dataframe(unsup["cluster_median_df"], use_container_width=True)

        st.write("**Cluster Feature Gaps (Top 5)**")
        st.dataframe(pd.DataFrame(unsup.get("cluster_feature_gaps", [])), use_container_width=True)

        if unsup["notes"]:
            st.write("**Notes**")
            for note in unsup["notes"]:
                st.write(f"- {note}")


def draw_llm_report(result: Optional[Dict[str, Any]]):
    if result is None:
        st.info("분석 실행 후 LLM 리포트를 생성할 수 있습니다.")
        return

    llm_report = st.session_state.get("llm_report")
    llm_error = st.session_state.get("llm_error")

    if llm_error:
        st.error(llm_error)

    if llm_report:
        st.markdown(llm_report)
    else:
        st.info("사이드바의 'LLM 리포트 생성' 버튼을 눌러 리포트를 생성하세요.")

    with st.expander("report_data (JSON)"):
        st.json(result["report_data"])


def main():
    st.set_page_config(page_title="Data Processer", layout="wide")
    chosen_font = configure_korean_matplotlib_font()
    st.title("Data Processer")
    st.caption("로컬 단일 실행용. 계산은 코드가 수행하고, LLM은 해석만 담당합니다.")

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "llm_report" not in st.session_state:
        st.session_state["llm_report"] = None
    if "llm_error" not in st.session_state:
        st.session_state["llm_error"] = None

    st.sidebar.header("설정")
    if chosen_font:
        st.sidebar.caption(f"한글 폰트 적용: {chosen_font}")
    else:
        st.sidebar.caption("한글 폰트를 찾지 못해 기본 폰트로 표시됩니다.")
    uploaded_file = st.sidebar.file_uploader("1) CSV 업로드", type=["csv"])
    mode = st.sidebar.radio("2) 분석 모드 선택", ["지도학습", "비지도학습"], index=0)

    available_cols: List[str] = []
    if uploaded_file is not None:
        available_cols = robust_read_columns(uploaded_file)

    target_col = None
    if mode == "지도학습":
        if available_cols:
            target_col = st.sidebar.selectbox("3) 타겟 컬럼 선택", available_cols)
        else:
            st.sidebar.warning("타겟 컬럼 목록을 불러오지 못했습니다. CSV 인코딩/구분자를 확인하세요.")

    st.sidebar.subheader("4) 옵션")
    max_rows = st.sidebar.number_input("max_rows 샘플링", min_value=1000, max_value=2_000_000, value=50_000, step=1000)
    test_size = st.sidebar.slider("test_size", min_value=0.05, max_value=0.4, value=0.2, step=0.05)
    random_seed = st.sidebar.number_input("random_seed", min_value=0, max_value=999999, value=42, step=1)
    outlier_option = st.sidebar.selectbox("이상치 처리", ["없음", "IQR 클리핑"], index=0)
    model_speed = st.sidebar.selectbox("모델링", ["빠름", "느림(CV 5-fold)"], index=0)

    run_clicked = st.sidebar.button("5) 분석 실행", type="primary")

    api_key_exists = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_exists:
        st.sidebar.caption("OPENAI_API_KEY 없음: LLM 리포트 기능 비활성화")

    can_run_llm = st.session_state.get("analysis_result") is not None and api_key_exists
    llm_clicked = st.sidebar.button("6) LLM 리포트 생성", disabled=not can_run_llm)

    if run_clicked:
        st.session_state["llm_report"] = None
        st.session_state["llm_error"] = None

        if uploaded_file is None:
            st.error("CSV 파일을 먼저 업로드하세요.")
        elif mode == "지도학습" and not target_col:
            st.error("지도학습 모드에서는 타겟 컬럼 선택이 필요합니다.")
        else:
            with st.spinner("분석 실행 중..."):
                try:
                    result = run_analysis(
                        uploaded_file=uploaded_file,
                        mode=mode,
                        target_col=target_col,
                        max_rows=int(max_rows),
                        test_size=float(test_size),
                        random_seed=int(random_seed),
                        outlier_option=outlier_option,
                        model_speed=model_speed,
                    )
                    st.session_state["analysis_result"] = result
                    st.success("분석 완료")
                except Exception as exc:
                    st.session_state["analysis_result"] = None
                    st.error(f"분석 실패: {exc}")

    if llm_clicked:
        result = st.session_state.get("analysis_result")
        if result is None:
            st.session_state["llm_error"] = "먼저 분석을 실행하세요."
        else:
            with st.spinner("LLM 리포트 생성 중..."):
                report, err = generate_llm_report(result["report_data"], result["df"].head(10))
                st.session_state["llm_report"] = report
                st.session_state["llm_error"] = err

    result = st.session_state.get("analysis_result")
    tab_labels = ["Overview", "EDA", "Preprocessing Log", "Modeling", "LLM Report"]
    active_tab = st.radio("탭", tab_labels, horizontal=True, key="active_tab")

    if result is None:
        if active_tab == "Overview":
            st.info("CSV 업로드 후 '분석 실행'을 눌러주세요.")
        elif active_tab == "LLM Report":
            st.info("분석 실행 후 LLM 리포트를 생성할 수 있습니다.")
        else:
            st.info("분석 실행 전입니다.")
        return

    if result["warnings"]:
        for w in result["warnings"]:
            st.warning(w)

    if active_tab == "Overview":
        draw_overview(result)
    elif active_tab == "EDA":
        draw_eda(result)
    elif active_tab == "Preprocessing Log":
        draw_preprocessing_log(result)
    elif active_tab == "Modeling":
        draw_modeling(result)
    else:
        draw_llm_report(result)


if __name__ == "__main__":
    main()
