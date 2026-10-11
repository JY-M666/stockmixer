# Source and scope

This repository contains the original StockMixer code from [SJTU-DMTai/StockMixer](https://github.com/SJTU-DMTai/StockMixer), `master` commit `cce13598afd3ff33ae317700a85ae08db0554652`.

The upstream `README.md`, `requirements.txt`, `src/train.py`, `src/load_data.py`, and `src/evaluator.py` are unchanged. In `src/model.py`, the original fixed downsample length `8` is replaced with `time_steps // 2` so a 60-day input works; for the original 16-day input this is still `8`. The new `src/qlib_adapter.py`, Alpha360 YAML, tests, and `ADAPTATION.md` support the ALSTM input format without altering the original NASDAQ training script.

The upstream datasets, paper, slides, images, IDE files, and compiled Python files are not included. To run the upstream training script, obtain the dataset from the original repository and place it in `dataset/` beside `src/`, then follow the original README. The training script's data paths are relative to `src/`.

The upstream repository does not contain a license file. This repository does not add a license or claim ownership of the upstream code.
