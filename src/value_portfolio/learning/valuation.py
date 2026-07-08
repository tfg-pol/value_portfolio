
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np

from value_portfolio.data.scores import ScoreRecord
from value_portfolio.learning.features import CrossSection, rank_normalize, scale_levels_by_assets

ModelName = Literal["ridge", "gbt"]


@dataclass(frozen=True)
class ValuationConfig:
    model: ModelName = "ridge"
    burn_in_sections: int = 96
    ridge_alpha: float = 1.0
    target: Literal["cap", "mb", "ma", "ret"] = "cap"
    scale_features: bool | None = None
    industry: bool = False
    shuffle_labels: bool = False
    seed: int = 0


def _make_model(config: ValuationConfig, n_features: int | None = None) -> Any:
    if config.model == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=config.ridge_alpha)
    from sklearn.ensemble import HistGradientBoostingRegressor

    categorical = None
    if config.industry and n_features is not None:
        # The industry code is the last column appended by `_design_matrix`.
        categorical = np.zeros(n_features, dtype=bool)
        categorical[-1] = True
    return HistGradientBoostingRegressor(random_state=config.seed, categorical_features=categorical)


def _design_matrix(section: CrossSection, config: ValuationConfig) -> np.ndarray:
    if config.model == "ridge":
        return rank_normalize(section.features)
    scale = config.scale_features
    if scale is None:
        scale = config.target in ("mb", "ma")
    features = scale_levels_by_assets(section.features) if scale else section.features
    if config.industry and section.industry is not None:
        features = np.column_stack([features, section.industry])
    return features


def fit_predict_expanding(
    sections: Sequence[CrossSection], config: ValuationConfig
) -> list[ScoreRecord]:
    if config.burn_in_sections < 1:
        raise ValueError(f"burn_in_sections must be >= 1, got {config.burn_in_sections}")
    if len(sections) <= config.burn_in_sections:
        return []

    if config.shuffle_labels:
        rng = np.random.default_rng(config.seed)
        targets = []
        for s in sections:
            permuted = s.target.copy()
            finite = np.flatnonzero(~np.isnan(permuted))
            permuted[finite] = permuted[finite][rng.permutation(finite.size)]
            targets.append(permuted)
    else:
        targets = [s.target for s in sections]

    records: list[ScoreRecord] = []
    for i in range(config.burn_in_sections, len(sections)):
        test = sections[i]
        train_x, train_y = [], []
        for j in range(i):
            design = _design_matrix(sections[j], config)
            labels = targets[j]
            keep = ~np.isnan(labels)
            train_x.append(design[keep])
            train_y.append(labels[keep])
        x_train = np.vstack(train_x)
        y_train = np.concatenate(train_y)
        model = _make_model(config, x_train.shape[1])
        model.fit(x_train, y_train)
        predicted = model.predict(_design_matrix(test, config))
        if config.target == "ret":
            score = predicted - predicted.mean()
        else:
            score = predicted - test.target
            score -= score.mean()
        for symbol, value in zip(test.symbols, score, strict=True):
            records.append(
                ScoreRecord(symbol=symbol, date=test.date, score=Decimal(f"{value:.6f}"))
            )
    return records


def write_scores_parquet(records: Sequence[ScoreRecord], path: Path | str) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "date": [rec.date.replace(tzinfo=None) for rec in records],
            "ticker": [rec.symbol for rec in records],
            "score": [float(rec.score) for rec in records],
        }
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
