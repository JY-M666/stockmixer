"""Train the original StockMixer on ALSTM Alpha360 inputs with fixed stock slots."""

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
POOL_FILE = Path(__file__).resolve().parents[1] / "configs" / "csi300_2014-12-31.txt"


def load_pool(path=POOL_FILE):
    stocks = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(stocks) != 300 or len(set(stocks)) != 300:
        raise ValueError("The fixed CSI300 pool must contain 300 unique stocks")
    return stocks


def align_day(features, labels, stocks):
    """Keep the same stock order on every day, with a mask for absent rows."""
    aligned = features.reindex(stocks)
    target = labels.reindex(stocks)
    valid = aligned.notna().all(axis=1).to_numpy() & target.notna().to_numpy()
    x = np.nan_to_num(aligned.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(target.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x, y[:, None], valid.astype(np.float32)[:, None]


def load_data(provider_uri, stocks, start, end):
    qlib.init(provider_uri=provider_uri, region="cn", kernels=1)
    handler = Alpha360(
        instruments=stocks,
        start_time=start,
        end_time=end,
        fit_start_time=start,
        fit_end_time=TRAIN_END,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[],
        label=["Ref($close, -2) / Ref($close, -1) - 1"],
    )
    features = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    labels = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]
    if features.shape[1] != 360:
        raise ValueError(f"Expected Alpha360 features, got {features.shape[1]}")
    return features, labels


def day_batches(features, labels, stocks, start, end):
    date_index = features.index.get_level_values("datetime")
    for date in date_index.unique().sort_values():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            x, y, mask = align_day(features.xs(date, level="datetime"), labels.xs(date, level="datetime"), stocks)
            if mask.any():
                yield date, x, y, mask


def run_epoch(model, batches, optimizer, device, alpha):
    training = optimizer is not None
    model.train(training)
    losses = []
    for _, x, y, mask in batches:
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        mask = torch.from_numpy(mask).to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            # Alpha360 is scaled by the current close; predict normalized next price.
            prediction = model(x)
            loss, _, _, _ = get_loss(prediction, y, torch.ones_like(y), mask, len(x), alpha)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
    if not losses:
        raise ValueError("No valid trading days in this segment")
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", required=True, help="Qlib daily CN data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default=TEST_END)
    parser.add_argument("--output", default="stockmixer_fixed_2014.pt")
    args = parser.parse_args()
    if pd.Timestamp(args.start) > pd.Timestamp(TRAIN_END) or pd.Timestamp(args.end) < pd.Timestamp("2017-01-01"):
        raise ValueError("The requested range must include the training and test periods")

    torch.manual_seed(12345678)
    np.random.seed(123456789)
    stocks = load_pool()
    features, labels = load_data(args.provider_uri, stocks, args.start, args.end)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockMixer(stocks=len(stocks), time_steps=60, channels=6, market=20, scale=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_valid = float("inf")
    for epoch in range(args.epochs):
        train = run_epoch(model, day_batches(features, labels, stocks, args.start, TRAIN_END), optimizer, device, 0.1)
        valid = run_epoch(model, day_batches(features, labels, stocks, "2015-01-01", VALID_END), None, device, 0.1)
        print(f"epoch {epoch + 1}: train loss {train:.6f}, valid loss {valid:.6f}", flush=True)
        if valid < best_valid:
            best_valid = valid
            torch.save({"model": model.state_dict(), "stocks": stocks, "pool_date": POOL_DATE}, args.output)
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test = run_epoch(model, day_batches(features, labels, stocks, "2017-01-01", args.end), None, device, 0.1)
    print(f"best valid loss {best_valid:.6f}, test loss {test:.6f}")


if __name__ == "__main__":
    main()
