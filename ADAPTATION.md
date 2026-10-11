# ALSTM 输入适配

这份适配使用你现有 ALSTM 的 Qlib Alpha360 输入：每只股票 6 个特征，每个特征回看 60 天，共 360 列。Alpha360 的列顺序是先给出第 1 个特征的 60 天，再给出第 2 个特征的 60 天；适配器将其转为官方 StockMixer 所需的 `[股票数, 60, 6]`。

官方模型的时间混合、指标混合、股票混合和两个输出分支均保留。`src/model.py` 只将下采样后的固定长度 8 改为 `time_steps // 2`，使 60 天输入的下采样长度为 30；原版 16 天输入仍得到 8。原始 `src/train.py` 不改，仍可用于官方 NASDAQ 数据。

CSI300 每日成分数会变化，而官方股票混合层需要固定股票数。适配器只用训练期建立股票位置；当日缺席股票的位置填零。训练期未见的新股票默认仅走官方模型的局部分支，并给出警告；设置 `unknown_stock: error` 可改为直接报错。完整股票截面的输出与官方模型一致。

## 运行

在已安装 Qlib、PyTorch 和项目依赖的环境中，从仓库根目录运行：

```bash
qrun examples/benchmarks/StockMixer/workflow_config_stockmixer_Alpha360.yaml
```

先将 YAML 中的 `provider_uri` 指向你的 Qlib 中国市场数据；CPU 运行时把 `GPU` 改为 `-1`。YAML 的数据处理、标签、训练/验证/测试日期和记录配置与本地 ALSTM 配置一致，模型参数为 `d_feat: 6`、`seq_len: 60`。

ALSTM 的标签经过 CSRankNorm，无法直接使用官方针对原始价格的 `get_loss`。Qlib 适配器对同一处理后标签使用 MSE，并仅用验证损失选择 checkpoint；官方 NASDAQ 脚本的损失函数保持不变。按日完整截面训练是股票混合层的输入要求。

若要改为 10 天窗口，还必须同时提供与之匹配的 6×10 特征列，并将 YAML 中的 `seq_len` 改为 10；只改 `seq_len` 不会自动裁剪 Alpha360 的 60 天数据。

## 2014 年末固定股票池实验

独立配置 `examples/benchmarks/StockMixer/workflow_config_stockmixer_Alpha360_fixed_20141231.yaml` 将 CSI300 股票池固定为 **2014-12-31** 的 300 只成分股。名单在 `universes/csi300_20141231.txt`；它由本地 Qlib 数据的 `instruments/csi300.txt` 重建，配置头部记录了该源文件的 SHA256。训练、验证、测试日期及其他模型参数与动态配置一致，`unknown_stock: error` 使任何池外股票直接报错。

如果训练机使用另一份 Qlib 数据，应先用那份数据重建名单，再运行固定池配置：

```bash
python scripts/build_fixed_universe.py --provider-uri /path/to/cn_data
qrun examples/benchmarks/StockMixer/workflow_config_stockmixer_Alpha360_fixed_20141231.yaml
```

同时确认生成的配置中 `qlib_init.provider_uri` 指向该数据目录。固定名单在训练、验证、测试期间不变，但部分股票可能尚未上市、停牌或退市，实际每天可用的股票行数仍会变化；适配器用零填补缺席股票的位置。基准仍为动态的 `SH000300` 指数。

此设计只避免使用 **2015–2020 年**成分信息决定股票池。用 2014 年末名单回选 **2008–2014 年**训练样本，仍存在训练期内的事后选股偏差；它也会改变 CSRankNorm 的每日截面。因此须单列为 **StockMixer under a fixed-universe setting**，不可称为与动态 CSI300 的 ALSTM 主实验完全同协议。

运行不依赖官方仓库的大型数据集。基础检查：

```bash
python -m unittest discover -s tests -p test_qlib_adapter.py
```
