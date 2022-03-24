from collections import defaultdict
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import wandb

plt.rc("text", usetex=True)

sns.set(style="white", font_scale=2)
plt.rcParams["font.size"] = 50


def plot_data(data, ax, label, n_ep=5, color="gray", stats_axis=0):
    mean = np.mean(data, axis=stats_axis)[:, -1]
    min_values = np.min(data, axis=stats_axis)[:, -1]
    max_values = np.max(data, axis=stats_axis)[:, -1]

    smooth_window = 10 if n_ep == 5 else 5
    mean = np.array(pd.Series(mean).rolling(smooth_window, min_periods=smooth_window).mean())
    min_values = np.array(pd.Series(min_values).rolling(smooth_window, min_periods=smooth_window).mean())
    max_values = np.array(pd.Series(max_values).rolling(smooth_window, min_periods=smooth_window).mean())

    steps = data[0, :, 0]
    ax.plot(steps, mean, "k", label=label, color=color)
    lb = min_values
    lb[lb < 0] = 0
    ax.fill_between(steps, max_values, min_values, color=color, alpha=0.10)
    return ax


# Linear interpolation between 2 datapoints
def interpolate(pt1, pt2, x):
    x1, y1 = pt1
    x2, y2 = pt2
    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return y


def merge_data(data, min_data_axs, n_datapoints, axis=0):
    """
    data:
        list, each element is a different seed
        data[i].shape = [n_evaluations, 2] (timestep, value)
    """
    step = min_data_axs // 1000
    idxs = np.arange(0, min_data_axs + step, step)
    n_pts = len(idxs)
    data_copy = []
    for d in data:
        d = np.array(d)
        run_values = np.zeros(shape=(n_pts, 2))
        interp = interp1d(d[:, axis], d[:, -1], kind="linear", fill_value=(0, d[:, -1][-1]), bounds_error=False)
        run_values[:, 1] = interp(idxs)
        run_values[:, 0] = idxs
        data_copy.append(run_values)

    data = np.stack(data_copy, axis=0)
    return data


# Data is a list
def plot_experiments(
    data,
    show=True,
    save=True,
    n_ep=5,
    save_name="return",
    save_folder="./analysis/figures/",
    x_lim=None,
    x_label="timesteps",
    y_label="success_Rate",
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharey=True)
    # ax.set_title("Evaluation")

    cm = plt.get_cmap("tab10")
    colors = cm(np.linspace(0, 1, len(data)))
    colors = [[0, 0, 1, 1], [1, 0, 0, 0.8]]
    for exp_data, c in zip(data, colors):
        name, data = exp_data
        ax = plot_data(data, ax, n_ep=n_ep, label=name, color=c, stats_axis=0)

    ax.set_xlabel(x_label.title())
    ax.set_ylabel(y_label.title())
    ax.set_xlim(xmin=0, xmax=x_lim)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_folder, "%s.png" % save_name), bbox_inches="tight", pad_inches=0)


class WandbPlots:
    def __init__(
        self,
        experiments: Dict,
        track_metrics: List[str],
        load_from_file: bool = True,
        show: bool = True,
        save_dir: str = "./analysis/figures",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = os.path.abspath(save_dir)
        json_filepath = os.path.join(self.save_dir, "exp_data.json")
        self.metrics = [m.split("/")[-1] for m in track_metrics]
        print("searching data in %s" % json_filepath)
        if os.path.isfile(json_filepath) and load_from_file:
            print("File found, loading data from previous wandb fetch")
            with open(json_filepath, "r") as outfile:
                data = json.load(outfile)
        else:
            print("No file found, loading data wandb..")
            data = self.read_from_wandb(experiments, track_metrics)
        self.data = data
        self.plot_stats(show)

    def read_from_wandb(self, experiments, wandb_metrics):
        # Get the runs
        _api = wandb.Api()
        runs = {}
        for exp_name, exp_id in experiments.items():
            runs[exp_name] = []
            for credentials in exp_id.keys():
                project_runs = ["%s/%s" % (credentials, run_id) for run_id in exp_id[credentials]]
                project_runs = [_api.run(name) for name in project_runs]
                runs[exp_name].extend(project_runs)
        # Get the data from the runs
        data = {}
        for exp_name, run_lst in runs.items():
            exp_data = defaultdict(list)
            for run in run_lst:
                run_data = defaultdict(list)  # metric: (timestep, episode, value)
                for row in run.scan_history():
                    if "eval_timestep" in row:
                        for metric in wandb_metrics:
                            if metric in row:
                                _row_data = [row["eval_timestep"], row["eval_episode"], row[metric]]
                                _metric = metric.split("/")[-1]
                                run_data[_metric].append(_row_data)
                # List of run data for each metric
                for metric in self.metrics:
                    exp_data[metric].append(run_data[metric])
            data[exp_name] = exp_data

        output_path = os.path.join(self.save_dir, "exp_data.json")
        with open(output_path, "w") as outfile:
            json.dump(data, outfile, indent=2)
        return data

    def plot_stats(self, show):
        for metric in self.metrics:
            metric_data = {
                "timesteps": {"data": [], "min_x_value": np.inf},
                "episodes": {"data": [], "min_x_value": np.inf},
            }

            for exp_name, exp_data in self.data.items():
                aligned_by_ep, aligned_by_ts, n_eval_ep = self.align_data(metric, exp_data[metric])
                metric_data["timesteps"]["data"].append([exp_name, aligned_by_ts])
                metric_data["episodes"]["data"].append([exp_name, aligned_by_ep])

                # Update crop values
                if aligned_by_ep[0][-1][0] < metric_data["episodes"]["min_x_value"]:
                    metric_data["episodes"]["min_x_value"] = aligned_by_ep[0][-1][0]

                if aligned_by_ts[0][-1][0] < metric_data["timesteps"]["min_x_value"]:
                    metric_data["timesteps"]["min_x_value"] = aligned_by_ts[0][-1][0]

            for x_label, plot_info in metric_data.items():
                # Crop to experiment with least episodes
                x_lim = plot_info["min_x_value"]
                save_name = "%s_by_%s" % (metric, x_label)
                plot_experiments(
                    plot_info["data"],
                    n_ep=n_eval_ep,
                    show=show,
                    save=True,
                    save_name=save_name,
                    save_folder=self.save_dir,
                    x_label=x_label,
                    x_lim=x_lim,
                    y_label="Success rate",
                )

    def align_data(self, metric, data):
        """
        data(list):
            contains as elements different run results
            of the same experiment
            len() = n_experiments
            - Each element is another list with columns:
                columns:[timesteps, episode, metric_value]
        """
        # Transform to success rate
        search_res = re.search(r"\((.*?)\)", metric)
        if search_res:
            search_res = search_res.group(1)
            n_eval_ep = int(search_res[:-2])  # Remove "ep"
        else:
            n_eval_ep = 5

        for i in range(len(data)):
            data[i] = np.array(data[i]).astype("float")
            data[i][:, -1] = data[i][:, -1] / n_eval_ep

        # Align to match timestep or episode log
        n_data_points = max([len(d) for d in data])
        ep_axis = 1
        min_data_axs = min([np.array(d)[:, ep_axis][-1] for d in data])
        data_by_ep = merge_data(data, min_data_axs, n_data_points, axis=ep_axis)

        ts_axis = 0
        min_data_axs = min([np.array(d)[:, ts_axis][-1] for d in data])
        min_data_axs = 400000
        data_by_ts = merge_data(data, min_data_axs, n_data_points, axis=ts_axis)
        return data_by_ep, data_by_ts, n_eval_ep


if __name__ == "__main__":
    # Tabletop Rand
    runs_orig = {
        "VAPO": {"jessibd/vapo_ablation": ["1yfwtff4", "2nxav2vp", "p7jxgqvu"]},
        "local-SAC": {"jessibd/vapo_ablation": ["3q2bo42g", "30z3jgi5", "1q43lu1l"]},
    }

    # Generalization
    runs_gen = {
        "VAPO": {"jessibd/vapo_ablation": ["2g3fqg16", "2p6qrqwq"]},
        "local-SAC": {"jessibd/vapo_ablation": ["194x7p41", "8898hvp5", "2mufff20"]},  # 194x7p41
    }

    run_info = {"vapo_gen_15objs": runs_gen, "vapo_15objs": runs_orig}

    metrics = ["eval/success(15ep)", "eval/success(5ep)"]
    for exp_name, runs in run_info.items():
        analysis = WandbPlots(
            runs, metrics, load_from_file=True, show=False, save_dir="./analysis/figures/%s" % exp_name
        )
