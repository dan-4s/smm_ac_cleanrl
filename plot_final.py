"""Script to generate final plots for papers."""

# Imports
from absl import app
from absl import flags
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import parse
import polars as pl

# Constants
BASELINES = ["SAC"]
TASKS = ["Hopper-v4", "Hopper-v4"] # ["Ant-v4", "Hopper-v4"] # The ant results are corrupted rn.


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
flags.DEFINE_string("folder", "final_results", "The results folder to put the plots in.")
def plot_data(_):
    folder = FLAGS.folder
    if(not os.path.exists(folder)):
        os.mkdir(folder)
    if(not os.path.exists(f"{folder}/plots")):
        os.mkdir(f"{folder}/plots")
    
    # Ensure that there exist results files under the results folder.
    for task in TASKS:
        if(len(glob.glob(f"{folder}/{task}*")) == 0):
            raise ValueError(f"There does not exist any results of the form: {folder}/{task}*")
    
    # Get ready to plot for each value of N.
    num_tasks = len(TASKS)
    num_cols = math.ceil(num_tasks / 2)
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_cols, sharey=True)
    fig.set_size_inches(5*num_cols, 5*num_rows)

    # Plot for each task, at a position in the axis.
    for idx, task in enumerate(TASKS):
        x_idx = idx % num_cols
        y_idx = idx // 2
        # Assume we always have at least 2 tasks.
        axis_idx = axs[x_idx, y_idx] if(num_tasks > 2) else axs[idx]
        # First, plot the baselines.
        for baseline in BASELINES:
            common_grid, mean_line, _, std_line, _ = process_flat_parquet(
                f"{folder}/{task}__{baseline}*/*.parquet")
            axis_idx.plot(common_grid, mean_line, label=baseline, marker="^", markevery=50)
            # axis_idx.fill_between(common_grid, mean_line-std_line, mean_line+std_line, alpha=0.2)
        
        # Then, plot all the relevant results by searching for all results.
        for result_folder in glob.glob(f"{folder}/{task}__SMM*"):
            parse_template = "final_results/{env}__SMM_lr={lr}_ref_freq={freq}_N={n}"
            SMM_hyperparams = parse.parse(parse_template, result_folder)
            lr, ref_freq, N = SMM_hyperparams["lr"], SMM_hyperparams["freq"], SMM_hyperparams["n"]
            common_grid, mean_line, _, std_line, _ = process_flat_parquet(
                f"{result_folder}/*.parquet")
            axis_idx.plot(common_grid, mean_line, label=f"SMM_lr={lr}_ref_freq={ref_freq}")
            # axis_idx.fill_between(common_grid, mean_line-std_line, mean_line+std_line, alpha=0.2)
    
        axis_idx.set_title(f"SMM-AC vs. SAC on {task}")
        axis_idx.set_xlabel("Steps")
        axis_idx.set_ylabel("Average episodic return")
        axis_idx.grid()
    axis_idx.legend(bbox_to_anchor=(1.05, 1)) # Use the last axis to put the legend.
    fig.savefig(f"{folder}/plots/combined_mujoco_results.png", bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    app.run(plot_data)

