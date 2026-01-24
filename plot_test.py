import polars as pl
import glob
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt

def process_nested_parquet(file_pattern, grid_step=100):
    # 1. Load and "Explode" the lists
    # include_file_paths identifies which of the 10 files each row came from
    lazy_df = (
        pl.scan_parquet(file_pattern, include_file_paths="run_id")
        .explode(["episodic_return", "episodic_length"])
    )
    
    df = lazy_df.collect().to_pandas()

    # 2. Setup the common grid for averaging
    # We want a shared X-axis from the earliest step to the latest
    min_s, max_s = df["episodic_step"].min(), df["episodic_step"].max()
    common_grid = np.arange(min_s, max_s, grid_step)

    # 3. Interpolate each run onto the grid
    run_results = []
    for rid in df["run_id"].unique():
        subset = df[df["run_id"] == rid].sort_values("episodic_step")
        
        # We map the irregular 'episodic_step' to our 'common_grid'
        interp_vals = np.interp(common_grid, subset["episodic_step"], subset["episodic_return"])
        
        run_results.append(pd.DataFrame({
            "step": common_grid,
            "return": interp_vals
        }))

    # 4. Aggregate across the 10 runs
    all_runs_df = pd.concat(run_results)
    final_df = all_runs_df.groupby("step")["return"].agg(["mean", "std"]).reset_index()

    # 5. Apply the EMA (0.99 coefficient -> alpha 0.01)
    final_df["ema"] = final_df["mean"].ewm(alpha=1-0.99).mean()
    
    return final_df

def process_flat_parquet(file_pattern, grid_points=1000):
    # 1. Load data - No explode needed!
    # Just select the columns we need
    df = (
        pl.scan_parquet(file_pattern, include_file_paths="run_id")
        .select(["episodic_step", "episodic_return", "run_id"])
        .collect()
        .to_pandas()
    )

    # 2. Create a common X-axis (Global Steps)
    # This creates a shared timeline for all runs to map onto
    common_grid = np.linspace(df["episodic_step"].min(), df["episodic_step"].max(), grid_points)

    # 3. Align all 10 runs to the common grid
    run_values = []
    for rid in df["run_id"].unique():
        run_subset = df[df["run_id"] == rid].sort_values("episodic_step")
        
        # Interpolate: Estimates the return at exactly the 'common_grid' steps
        interp_y = np.interp(common_grid, run_subset["episodic_step"], run_subset["episodic_return"])
        run_ema = pd.Series(interp_y).ewm(alpha=0.01).mean().values
        run_values.append(run_ema)

    # 4. Aggregate
    # Stack into a matrix of (Number of Runs, Grid Points)
    matrix = np.stack(run_values) 
    mean_line = np.mean(matrix, axis=0)
    median_line = np.median(matrix, axis=0)
    std_line = np.std(matrix, axis=0)
    
    # 5. EMA (Smoothing)
    # 0.99 smoothing coefficient translates to alpha = 0.01
    # ema_line = pd.Series(mean_line).ewm(alpha=0.01).mean().values

    return common_grid, mean_line, median_line, std_line

if __name__ == "__main__":
   # Usage
    common_grid, mean_line, median_line, std_line = process_flat_parquet("results_january_23/Hopper-v4__SMM_lr=5e-6_ref_freq=6_N=5/*.parquet")
    plt.plot(common_grid, mean_line, label="mean of EMAs")
    plt.plot(common_grid, median_line, label="median of EMAs")
    plt.fill_between(common_grid, mean_line-std_line, mean_line+std_line, alpha=0.2)
    # tab1 = pq.read_table("test_results_1.parquet")
    # plt.plot(tab1["episodic_step"].to_pylist(), tab1["episodic_return"].to_pylist(), label="line1")
    # tab2 = pq.read_table("test_results_2.parquet")
    # plt.plot(tab2["episodic_step"].to_pylist(), tab2["episodic_return"].to_pylist(), label="line2")
    plt.legend()
    plt.savefig("test_results.png")
