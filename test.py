import time
import numpy as np
import pandas as pd
from autotau.core.parallel import ParallelCyclesAutoTauFitter

# === 1. 加载数据 ===
path = '/home/lidonghaowsl/develop/Minitest-OECT-dataprocessing/output.parquet'
print("Loading parquet...")
load_start = time.time()

df = pd.read_parquet(path, engine="pyarrow")
print(f"Loaded dataframe with columns {list(df.columns)} and {len(df)} rows in {time.time() - load_start:.2f}s")

continuous_time = df["continuous_time"].to_numpy()
signal = df["drain_current"].to_numpy()
del df  # 释放内存

# === 2. 采样与周期设置 ===
sample_step = float(np.median(np.diff(continuous_time[:1000])))
sample_rate = 1.0 / sample_step
period = 0.25

print(f"Sample step: {sample_step}, sample rate: {sample_rate}, total duration: {continuous_time[-1] - continuous_time[0]:.2f}s")

# === 3. 初始化并行 Tau 拟合器 ===
fitter = ParallelCyclesAutoTauFitter(
    continuous_time,
    signal,
    period=period,
    sample_rate=sample_rate,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=25,
    window_start_idx_step=1,
    normalize=False,
    language="cn",
    show_progress=True,
    max_workers=96,
)

# === 4. 拟合所有周期 ===
start = time.time()
print("Fitting all cycles...")
fitter.fit_all_cycles(interp=True, points_after_interp=100, r_squared_threshold=0.95)
elapsed = time.time() - start
print(f"fit_all_cycles completed in {elapsed/60:.2f} minutes")

# === 5. 输出结果汇总 ===
summary = fitter.get_summary_data()
if summary is not None:
    print(summary.head())
    desc = summary[["tau_on", "tau_off", "r_squared_on", "r_squared_off"]].describe()
    print(desc)
    print(f"Total cycles: {len(summary)}, refitted cycles: {len(fitter.refitted_cycles)}")