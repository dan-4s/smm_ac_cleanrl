"""Plotting script for local parquet files."""

# Imports
from absl import app
from absl import flags
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import polars as pl

# Constants
PLOT_N = [2, 5]
PLOT_LR = ["1e-6","5e-6", "1e-5", "5e-5"]
PLOT_REF_FREQ = [2, 4, 6]
BASELINES = ["SAC"]


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

    return common_grid, mean_line, median_line, std_line, matrix


# Define the flags for command line input.
FLAGS = flags.FLAGS
flags.DEFINE_string("folder", "results", "The results folder to put the plots in.")
flags.DEFINE_string("task", "Hopper-v4", "The task for which to compile graphs.")
def plot_data(_):
    folder = FLAGS.folder
    if(not os.path.exists(folder)):
        os.mkdir(folder)
    if(not os.path.exists(f"{folder}/plots")):
        os.mkdir(f"{folder}/plots")
    
    # Ensure that there exist results files under the results folder.
    task = FLAGS.task
    if(len(glob.glob(f"{folder}/{task}*")) == 0):
        raise ValueError(f"There does not exist any results of the form: {folder}/{task}*")
    
    # Get ready to plot for each value of N.
    num_plots = len(PLOT_N)
    fig, axs = plt.subplots(1, num_plots, sharey=True)
    fig.set_size_inches(5*num_plots, 5)

    # Plot for each N.
    for idx, N in enumerate(PLOT_N):
        axis_idx = axs[idx] if(num_plots > 1) else axs
        # First, plot the baselines.
        for baseline in BASELINES:
            if(baseline == "SAC"):
                baseline = f"SAC_N={N}"
            common_grid, mean_line, _, std_line, _ = process_flat_parquet(
                f"{folder}/{task}__{baseline}/*.parquet")
            axis_idx.plot(common_grid, mean_line, label=baseline, marker="^", markevery=50)
            # axis_idx.fill_between(common_grid, mean_line-std_line, mean_line+std_line, alpha=0.2)
        
        # Then, plot all the relevant results.
        for lr in PLOT_LR:
            for ref_freq in PLOT_REF_FREQ:
                common_grid, mean_line, _, std_line, _ = process_flat_parquet(
                    f"{folder}/{task}__SMM_lr={lr}_ref_freq={ref_freq}_N={N}/*.parquet")
                axis_idx.plot(common_grid, mean_line, label=f"SMM_lr={lr}_ref_freq={ref_freq}")
                # axis_idx.fill_between(common_grid, mean_line-std_line, mean_line+std_line, alpha=0.2)
    
        axis_idx.set_title(f"SMM-AC vs. SAC with N={N} on Hopper-v4")
        axis_idx.set_xlabel("Steps")
        axis_idx.set_ylabel("Average episodic return")
        axis_idx.grid()
    axis_idx.legend(bbox_to_anchor=(1.05, 1)) # Use the last axis to put the legend.
    fig.savefig(f"{folder}/plots/{task}.png", bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    app.run(plot_data)

