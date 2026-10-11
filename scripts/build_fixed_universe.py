"""Freeze a Qlib CSI300 universe at the training end date.

Run from the repository root with the same Qlib data source used for training.
"""

import argparse
import hashlib
from pathlib import Path

import qlib
from qlib.data import D


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "examples/benchmarks/StockMixer/workflow_config_stockmixer_Alpha360.yaml"
FIXED_CONFIG = ROOT / "examples/benchmarks/StockMixer/workflow_config_stockmixer_Alpha360_fixed_20141231.yaml"
UNIVERSE = ROOT / "universes/csi300_20141231.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", required=True, help="Qlib data directory used for the experiment")
    parser.add_argument("--output", type=Path, default=FIXED_CONFIG)
    parser.add_argument("--universe-output", type=Path, default=UNIVERSE)
    args = parser.parse_args()

    as_of = "2014-12-31"
    provider = Path(args.provider_uri).expanduser().resolve()
    qlib.init(provider_uri=str(provider), region="cn")
    calendar = D.calendar(start_time=as_of, end_time=as_of)
    if len(calendar) != 1 or str(calendar[0].date()) != as_of:
        raise ValueError("Snapshot date is not a Qlib trading day: %s" % as_of)
    codes = sorted(set(D.list_instruments(D.instruments("csi300"),
                                          start_time=as_of, end_time=as_of, as_list=True)))
    if len(codes) != 300:
        raise ValueError("Expected 300 CSI300 members on %s; found %d" % (as_of, len(codes)))

    source = provider / "instruments/csi300.txt"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else "unavailable"

    base = BASE_CONFIG.read_text(encoding="utf-8")
    market_line = "market: &market csi300"
    policy_line = "unknown_stock: local"
    if base.count(market_line) != 1 or base.count(policy_line) != 1:
        raise ValueError("Base config changed; inspect the replacements before regenerating")
    fixed = base.replace(market_line, "market: &market\n" + "\n".join("    - " + code for code in codes))
    fixed = fixed.replace(policy_line, "unknown_stock: error")
    header = ("# CSI300 membership frozen at %s from the Qlib provider.\n"
              "# Source instruments/csi300.txt SHA256: %s\n"
              "# Rebuild with scripts/build_fixed_universe.py if the provider changes.\n"
              % (as_of, source_hash))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + fixed, encoding="utf-8")
    args.universe_output.parent.mkdir(parents=True, exist_ok=True)
    args.universe_output.write_text("\n".join(codes) + "\n", encoding="utf-8")

    print("date=%s members=%d source_sha256=%s" % (as_of, len(codes), source_hash))
    print("config=%s" % args.output.resolve())
    print("universe=%s" % args.universe_output.resolve())


if __name__ == "__main__":
    main()
