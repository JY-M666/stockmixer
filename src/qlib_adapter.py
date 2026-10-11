"""Feed Qlib Alpha360 features to the original StockMixer architecture.

The original NASDAQ training script remains available as ``src/train.py``.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path

from src.model import StockMixer as OriginalStockMixer


class DailyStockMixer(nn.Module):
    """Use the original layers with a fixed train-only stock vocabulary."""

    def __init__(self, stocks, lookback=60, features=6, market=20):
        super().__init__()
        self.lookback, self.features = lookback, features
        self.core = OriginalStockMixer(stocks, lookback, features, market, scale=3)

    def forward(self, flat, slots):
        # Alpha360 columns are feature-major: 60 days of feature 1, then feature 2.
        sequence = flat.reshape(-1, self.features, self.lookback).transpose(1, 2)
        coarse = self.core.conv(sequence.transpose(1, 2)).transpose(1, 2)
        encoded = self.core.channel_fc(self.core.mixer(sequence, coarse)).squeeze(-1)
        local = self.core.time_fc(encoded).squeeze(-1)

        known = slots >= 0
        # Absent stocks have zero encoded values, including when encoder layers have biases.
        panel = encoded.new_zeros((self.core.stock_mixer.dense1.in_features, encoded.shape[1]))
        panel = panel.index_copy(0, slots[known], encoded[known])
        market = self.core.stock_mixer(panel)
        addition = self.core.time_fc_(market).squeeze(-1)
        return local.index_add(0, torch.where(known)[0], addition[slots[known]])


class StockMixerQlib(Model):
    """Qlib train/predict interface for the ALSTM Alpha360 input protocol."""

    def __init__(self, d_feat=6, seq_len=60, market_num=20, n_epochs=200,
                 lr=1e-3, early_stop=20, GPU=0, seed=0, unknown_stock="local"):
        if min(d_feat, market_num, n_epochs, early_stop) < 1 or seq_len < 2 or lr <= 0:
            raise ValueError("Dimensions, epochs, patience and lr must be positive; seq_len >= 2.")
        if unknown_stock not in ("local", "error"):
            raise ValueError("unknown_stock must be 'local' or 'error'.")
        self.d_feat, self.seq_len, self.market_num = d_feat, seq_len, market_num
        self.n_epochs, self.lr, self.early_stop = n_epochs, lr, early_stop
        self.seed, self.unknown_stock = seed, unknown_stock
        self.device = torch.device("cuda:%d" % GPU if GPU >= 0 and torch.cuda.is_available() else "cpu")
        self.logger = get_module_logger("StockMixerQlib")
        self.fitted = False

    def _check(self, features):
        if not isinstance(features.index, pd.MultiIndex) or set(features.index.names) != {"datetime", "instrument"}:
            raise ValueError("Expected a MultiIndex named datetime and instrument.")
        if not features.index.is_unique:
            raise ValueError("Duplicate datetime/instrument rows are not supported.")
        if features.shape[1] != self.d_feat * self.seq_len:
            raise ValueError("Feature width must equal d_feat * seq_len.")
        if not np.isfinite(features.to_numpy(dtype=np.float32)).all():
            raise ValueError("Non-finite features: use the ALSTM Fillna preprocessing.")

    def _days(self, features, labels=None):
        self._check(features)
        slots = self.instruments_.get_indexer(features.index.get_level_values("instrument"))
        unknown = slots < 0
        if unknown.any():
            count = int(unknown.sum())
            if self.unknown_stock == "error":
                raise ValueError("%d rows contain stocks absent from the training universe." % count)
            warnings.warn("StockMixer: %d/%d rows use the local branch for unseen stocks."
                          % (count, len(slots)), UserWarning, stacklevel=2)
        values = features.to_numpy(dtype=np.float32)
        targets = None if labels is None else labels.to_numpy(dtype=np.float32).reshape(-1)
        groups = features.groupby(level="datetime", sort=True).indices
        return [(positions, values[positions], slots[positions],
                 None if targets is None else targets[positions]) for positions in groups.values()]

    def _predict_day(self, day):
        _, features, slots, _ = day
        return self.model_(torch.as_tensor(features, device=self.device),
                           torch.as_tensor(slots, dtype=torch.long, device=self.device))

    def _evaluate(self, days):
        self.model_.eval()
        squared, count = 0.0, 0
        with torch.no_grad():
            for day in days:
                target = torch.as_tensor(day[3], device=self.device)
                valid = torch.isfinite(target)
                if valid.any():
                    squared += ((self._predict_day(day)[valid] - target[valid]) ** 2).sum().item()
                    count += int(valid.sum())
        if not count or not np.isfinite(squared):
            raise ValueError("No finite validation loss; check labels and training stability.")
        return squared / count

    def fit(self, dataset, evals_result=None, save_path=None):
        self.fitted = False
        train, valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"],
                                       data_key=DataHandlerLP.DK_L)
        if train.empty or valid.empty:
            raise ValueError("Training and validation data must be nonempty.")
        if train["label"].shape[1] != 1 or valid["label"].shape[1] != 1:
            raise ValueError("StockMixer supports one label column.")
        self._check(train["feature"])
        self.feature_columns_ = train["feature"].columns.copy()
        if not valid["feature"].columns.equals(self.feature_columns_):
            raise ValueError("Validation feature columns/order differ from training.")
        if train.index.get_level_values("datetime").max() >= valid.index.get_level_values("datetime").min():
            raise ValueError("Training dates must precede validation dates.")
        self.instruments_ = pd.Index(sorted(train.index.get_level_values("instrument").unique()))
        train_days = self._days(train["feature"], train["label"])
        valid_days = self._days(valid["feature"], valid["label"])
        rng = np.random.RandomState(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.model_ = DailyStockMixer(len(self.instruments_), self.seq_len, self.d_feat, self.market_num).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        result = evals_result if evals_result is not None else {}
        result["train"], result["valid"] = [], []
        best, best_state, stale = np.inf, None, 0
        for epoch in range(self.n_epochs):
            self.model_.train()
            for index in rng.permutation(len(train_days)):
                day = train_days[index]
                target = torch.as_tensor(day[3], device=self.device)
                finite = torch.isfinite(target)
                if not finite.any():
                    continue
                optimizer.zero_grad()
                loss = ((self._predict_day(day)[finite] - target[finite]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
            train_loss, valid_loss = self._evaluate(train_days), self._evaluate(valid_days)
            result["train"].append(-train_loss)
            result["valid"].append(-valid_loss)
            self.logger.info("epoch %d train MSE %.6f valid MSE %.6f", epoch, train_loss, valid_loss)
            if valid_loss < best:
                best, stale = valid_loss, 0
                self.best_epoch_ = epoch
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                stale += 1
                if stale >= self.early_stop:
                    break
        if best_state is None:
            raise ValueError("No valid training checkpoint.")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        torch.save({"state_dict": best_state, "instruments": self.instruments_.tolist(),
                    "feature_columns": self.feature_columns_.tolist(), "d_feat": self.d_feat,
                    "seq_len": self.seq_len, "market_num": self.market_num,
                    "unknown_stock": self.unknown_stock}, get_or_create_path(save_path))
        self.fitted = True

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("StockMixer is not fitted.")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if not features.columns.equals(self.feature_columns_):
            raise ValueError("Prediction feature columns/order differ from training.")
        days = self._days(features)
        predictions = np.empty(len(features), dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            for day in days:
                predictions[day[0]] = self._predict_day(day).cpu().numpy()
        return pd.Series(predictions, index=features.index)
