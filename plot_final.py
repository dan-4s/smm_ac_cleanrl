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
BASELINES = ["SAC", "TD3", "PPO"]
BASELINE_TO_MARKER = ["^", "o", ".", "o"]
SMM_MARKER = "*"
TASKS = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Pusher-v4", "Walker2d-v4"]
# TASKS = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Pusher-v4", "Swimmer-v4", "Walker2d-v4"]

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
    fig, axs = plt.subplots(num_rows, num_cols, sharey=False)
    fig.set_size_inches(5*num_cols, 5*num_rows)

    # Plot for each task, at a position in the axis.
    for idx, task in enumerate(TASKS):
        x_idx = idx % num_cols
        y_idx = idx // num_cols
        print((x_idx, y_idx, num_cols, num_tasks, task))
        # Assume we always have at least 2 tasks.
        axis_idx = axs[y_idx, x_idx] if(num_tasks > 2) else axs[idx]
        # First, plot the baselines.
        for base_idx, baseline in enumerate(BASELINES):
            common_grid, mean_line, _, std_line, _ = process_flat_parquet(
                f"{folder}/{task}__{baseline}*/*.parquet")
            axis_idx.plot(
                common_grid,
                mean_line,
                label=baseline,
                marker=BASELINE_TO_MARKER[base_idx],
                markevery=50,
            )
            axis_idx.fill_between(
                common_grid,
                mean_line-std_line,
                mean_line+std_line,
                alpha=0.2,
            )
        
        # Then, plot all the relevant results by searching for all results.
        for result_folder in glob.glob(f"{folder}/{task}__SMM*"):
            parse_template = "final_results/{env}__SMM_lr={lr}_ref_freq={freq}"
            SMM_hyperparams = parse.parse(parse_template, result_folder)
            lr, ref_freq = SMM_hyperparams["lr"], SMM_hyperparams["freq"]
            common_grid, mean_line, _, std_line, _ = process_flat_parquet(
                f"{result_folder}/*.parquet")
            axis_idx.plot(
                common_grid,
                mean_line,
                label=f"SMM",
                marker=SMM_MARKER,
                markevery=50,
            ) # Since we'll just plot the best SMM result.
            # axis_idx.plot(common_grid, mean_line, label=f"SMM_lr={lr}_ref_freq={ref_freq}")
            axis_idx.fill_between(
                common_grid,
                mean_line-std_line,
                mean_line+std_line,
                alpha=0.2,
            )
    
        axis_idx.set_title(f"SMM-AC vs. baselines on {task}")
        axis_idx.set_xlabel("Steps")
        axis_idx.set_ylabel("Average episodic return")
        axis_idx.grid()
    
    # If there are an odd number of tasks, put the legend in the last subplot, else, put it to the side.
    if(num_tasks % 2 != 0):
        legend_ax = axs[1, -1]

        # Clear the target subplot and turn off its axes.
        legend_ax.clear()
        legend_ax.set_axis_off()

        # Manually create legend handles (patches or lines) for all plots.
        # This is necessary because the legend is being created in an empty axis
        handles, labels = axis_idx.get_legend_handles_labels()

        # 5. Add the combined legend to the target subplot
        legend_ax.legend(handles=handles, labels=labels, loc='center', fancybox=True, shadow=True, ncol=1)
    else:
        axis_idx.legend(bbox_to_anchor=(1.05, 1)) # Use the last axis to put the legend.
    fig.savefig(f"{folder}/plots/combined_mujoco_results.png", bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    app.run(plot_data)

