import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.qlib_adapter import DailyStockMixer, StockMixerQlib


class SyntheticDataset:
    def __init__(self):
        self.calls = []
        rng = np.random.RandomState(10)
        self.frames = {}
        for segment, dates in (("train", ["2010-01-01", "2010-01-02"]),
                               ("valid", ["2011-01-01"]), ("test", ["2012-01-01"])):
            index = pd.MultiIndex.from_product([pd.to_datetime(dates), ["A", "B"]],
                                               names=["datetime", "instrument"])
            feature = pd.DataFrame(rng.randn(len(index), 360).astype(np.float32), index=index)
            label = pd.DataFrame(rng.randn(len(index), 1).astype(np.float32), index=index)
            self.frames[segment] = pd.concat({"feature": feature, "label": label}, axis=1)

    def prepare(self, segments, col_set, data_key):
        self.calls.append((segments, col_set, data_key))
        if isinstance(segments, list):
            return [self.frames[s][col_set] for s in segments]
        return self.frames[segments][col_set]


class StockMixerInputTests(unittest.TestCase):
    def test_full_panel_matches_original_and_supports_60_days(self):
        torch.manual_seed(1)
        net = DailyStockMixer(stocks=3, lookback=60, features=6, market=2)
        flat = torch.randn(3, 360)
        slots = torch.arange(3)
        actual = net(flat, slots)
        original = net.core(flat.reshape(3, 6, 60).transpose(1, 2)).squeeze(-1)
        torch.testing.assert_close(actual, original)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in net.parameters()))

    def test_unknown_stock_uses_local_branch(self):
        net = DailyStockMixer(stocks=3, lookback=60, features=6, market=2)
        flat = torch.randn(2, 360)
        slots = torch.tensor([1, -1])
        result = net(flat, slots)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[0], net(flat[:1], slots[:1])[0])

    def test_fit_reads_only_train_valid_and_preserves_prediction_index(self):
        dataset = SyntheticDataset()
        model = StockMixerQlib(n_epochs=1, market_num=2, GPU=-1)
        with tempfile.TemporaryDirectory() as tmp:
            model.fit(dataset, save_path=str(Path(tmp) / "model.pt"))
            self.assertEqual(dataset.calls, [(["train", "valid"], ["feature", "label"], "learn")])
            prediction = model.predict(dataset)
            self.assertTrue(prediction.index.equals(dataset.frames["test"].index))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue((Path(tmp) / "model.pt").exists())

    def test_config_uses_alstm_input(self):
        config = Path(__file__).resolve().parents[1] / "examples/benchmarks/StockMixer/workflow_config_stockmixer_Alpha360.yaml"
        task = yaml.safe_load(config.read_text(encoding="utf-8"))["task"]
        self.assertEqual(task["model"]["module_path"], "src.qlib_adapter")
        self.assertEqual(task["model"]["kwargs"]["d_feat"], 6)
        self.assertEqual(task["model"]["kwargs"]["seq_len"], 60)
        self.assertEqual(task["dataset"]["kwargs"]["handler"]["class"], "Alpha360")

    def test_fixed_universe_config_changes_only_the_universe_and_policy(self):
        root = Path(__file__).resolve().parents[1]
        config_dir = root / "examples/benchmarks/StockMixer"
        dynamic = yaml.safe_load((config_dir / "workflow_config_stockmixer_Alpha360.yaml").read_text(encoding="utf-8"))
        fixed = yaml.safe_load((config_dir / "workflow_config_stockmixer_Alpha360_fixed_20141231.yaml").read_text(encoding="utf-8"))
        codes = (root / "universes/csi300_20141231.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(codes), 300)
        self.assertEqual(len(set(codes)), 300)
        self.assertEqual(codes, sorted(codes))
        self.assertEqual(fixed["market"], codes)
        handler = fixed["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
        self.assertEqual(handler["instruments"], codes)
        self.assertEqual(fixed["task"]["model"]["kwargs"]["unknown_stock"], "error")
        dynamic.pop("market")
        fixed.pop("market")
        dynamic["task"]["dataset"]["kwargs"]["handler"]["kwargs"].pop("instruments")
        fixed["task"]["dataset"]["kwargs"]["handler"]["kwargs"].pop("instruments")
        dynamic["task"]["model"]["kwargs"].pop("unknown_stock")
        fixed["task"]["model"]["kwargs"].pop("unknown_stock")
        self.assertEqual(dynamic, fixed)


if __name__ == "__main__":
    unittest.main()
