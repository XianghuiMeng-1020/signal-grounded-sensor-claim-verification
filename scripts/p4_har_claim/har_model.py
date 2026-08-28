"""Standard Random Forest HAR. Hyperparameters frozen; no search."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS, SEED
from .dictionary import family_of
from .features import extract_features, feature_names


def _order(windows: list[dict]) -> list[str]:
    names = []
    seen = set()
    for w in windows:
        for k in w["available_channels"]:
            if k not in seen:
                seen.add(k)
                names.append(k)
    return names


def fit_rf(train_windows: list[dict]) -> tuple[RandomForestClassifier, list[str]]:
    order = _order(train_windows)
    X = np.stack([extract_features(w["channels"], float(w["fs"]), order) for w in train_windows])
    y = np.array([str(w["activity"]) for w in train_windows], dtype=object)
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf, order


def predict_activity(clf: RandomForestClassifier, order: list[str], window_or_item: dict) -> str:
    x = extract_features(window_or_item["channels"], float(window_or_item["fs"]), order).reshape(1, -1)
    return str(clf.predict(x)[0])


def predict_families(clf: RandomForestClassifier, order: list[str], items: list[dict]) -> dict[str, str | None]:
    out = {}
    for it in items:
        act = predict_activity(clf, order, it)
        out[it["item_id"]] = family_of(it["dataset"], act)
    return out


def predict_activities(clf: RandomForestClassifier, order: list[str], windows: list[dict]) -> dict[str, str]:
    return {w["window_id"]: predict_activity(clf, order, w) for w in windows}


def model_card(order: list[str]) -> dict:
    return {
        "architecture": "RandomForestClassifier",
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": RF_MAX_DEPTH,
        "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "random_state": SEED,
        "features": feature_names(order),
        "training": "subject-grouped existing development split; unused labeled windows only",
    }
