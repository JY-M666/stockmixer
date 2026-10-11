"""Train StockMixer on 60x6 Alpha360 inputs with fixed CSI300 stock slots."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset.handler import DataHandlerLP

from model import StockMixer, get_loss


POOL_DATE = "2014-12-31"
TRAIN_END = "2014-12-31"
VALID_END = "2016-12-31"
TEST_END = "2020-08-01"


def fixed_stocks(provider_uri):
    path = Path(provider_uri).expanduser() / "instruments" / "csi300.txt"
    stocks = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stock, first, last = line.split()
        if first <= POOL_DATE <= last:
            stocks.add(stock)
    if len(stocks) != 300:
        raise ValueError("Expected 300 CSI300 constituents on %s, got %d" % (POOL_DATE, len(stocks)))
    return sorted(stocks)


def load_qlib(provider_uri, stocks):
    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region="cn", kernels=1)
    handler = Alpha360(
        instruments=stocks,
        start_time="2008-01-01",
        end_time=TEST_END,
        fit_start_time="2008-01-01",
        fit_end_time=TRAIN_END,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[],
        # Match official StockMixer: input through t, predict the t -> t+1 return.
        label=["Ref($close, -1) / $close - 1", "$close"],
    )
    features = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    raw_features = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    targets = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I)
    if features.shape[1] != 360:
        raise ValueError("Expected 360 Alpha360 features, got %d" % features.shape[1])
    return features, raw_features, targets


def day_batch(date, stocks, features, raw_features, targets):
    x = features.xs(date, level="datetime").reindex(stocks)
    raw = raw_features.xs(date, level="datetime").reindex(stocks)
    target = targets.xs(date, level="datetime").reindex(stocks)
    values = x.to_numpy(dtype=np.float32)
    raw_close = raw.iloc[:, :60].to_numpy(dtype=np.float32)
    returns = target.iloc[:, 0].to_numpy(dtype=np.float32)
    prices = target.iloc[:, 1].to_numpy(dtype=np.float32)
    valid = (
        np.isfinite(raw_close).all(axis=1)
        & np.isfinite(values).all(axis=1)
        & np.isfinite(returns)
        & np.isfinite(prices)
        & (prices > 0)
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values[~valid] = 0.0
    returns = np.where(valid, returns, 0.0)[:, None]
    prices = np.where(valid, prices, 1.0)[:, None]
    return values, returns, prices, valid.astype(np.float32)[:, None]


def run_epoch(model, dates, stocks, features, raw_features, targets, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses = []
    counts = []
    for date in dates:
        x, target, base_price, mask = day_batch(date, stocks, features, raw_features, targets)
        if not mask.any():
            continue
        x, target, base_price, mask = (
            torch.from_numpy(array).to(device) for array in (x, target, base_price, mask)
        )
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            # Alpha360 has no absolute price scale; restore it before the official loss.
            predicted_price = model(x) * base_price
            loss, _, _, _ = get_loss(predicted_price, target, base_price, mask, len(stocks), 0.1)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        counts.append(mask.sum().item())
    if not losses:
        raise ValueError("No valid trading days in this segment")
    return float(np.mean(losses)), float(np.mean(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="stockmixer_fixed.pt")
    args = parser.parse_args()

    np.random.seed(123456789)
    torch.manual_seed(12345678)
    stocks = fixed_stocks(args.provider_uri)
    features, raw_features, targets = load_qlib(args.provider_uri, stocks)
    dates = features.index.get_level_values("datetime").unique().sort_values()
    train_dates = dates[(dates >= pd.Timestamp("2008-01-01")) & (dates <= pd.Timestamp(TRAIN_END))].to_numpy()
    valid_dates = dates[(dates >= pd.Timestamp("2015-01-01")) & (dates <= pd.Timestamp(VALID_END))]
    test_dates = dates[(dates >= pd.Timestamp("2017-01-01")) & (dates <= pd.Timestamp(TEST_END))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockMixer(stocks=300, time_steps=60, channels=6, market=20, scale=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_valid = float("inf")
    for epoch in range(args.epochs):
        np.random.shuffle(train_dates)
        train_loss, train_count = run_epoch(model, train_dates, stocks, features, raw_features, targets, device, optimizer)
        valid_loss, valid_count = run_epoch(model, valid_dates, stocks, features, raw_features, targets, device)
        test_loss, test_count = run_epoch(model, test_dates, stocks, features, raw_features, targets, device)
        print("epoch %d: train %.6f (%0.1f/300), valid %.6f (%0.1f/300), test %.6f (%0.1f/300)"
              % (epoch + 1, train_loss, train_count, valid_loss, valid_count, test_loss, test_count), flush=True)
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save({"model": model.state_dict(), "stocks": stocks, "pool_date": POOL_DATE,
                        "epoch": epoch + 1}, args.output)
    print("best validation loss: %.6f" % best_valid)


if __name__ == "__main__":
    main()
