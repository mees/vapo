import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns

plt.rc("text", usetex=True)

sns.set(style="white", font_scale=2)
# plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
plt.rcParams["font.size"] = 24


def plot_data(data, ax, label, n_ep=6, color="gray", stats_axis=0):
    mean = np.mean(data, axis=stats_axis)[:, -1]
    # std = np.std(data, axis=stats_axis)[:, -1]
    min_values = np.min(data, axis=stats_axis)[:, -1]
    max_values = np.max(data, axis=stats_axis)[:, -1]

    smooth_window = 5
    mean = np.array(pd.Series(mean).rolling(smooth_window, min_periods=smooth_window).mean())
    # std = np.array(pd.Series(std).rolling(smooth_window,
    #                                       min_periods=smooth_window).mean())
    min_values = np.array(pd.Series(min_values).rolling(smooth_window, min_periods=smooth_window).mean())
    max_values = np.array(pd.Series(max_values).rolling(smooth_window, min_periods=smooth_window).mean())

    steps = data[0, :, 0]
    # for learning_curve in data:
    #     smooth_data = np.array(pd.Series(learning_curve[:, 1]).rolling(smooth_window,
    #                                             min_periods=smooth_window).mean())
    #     ax.plot(learning_curve[:, 0], smooth_data, 'k', linewidth=1, color=color)

    ax.plot(steps, mean, "k", label=label, color=color)
    # lb = mean - std
    lb = min_values
    lb[lb < 0] = 0
    # ax.fill_between(steps, mean + std, lb, color=color, alpha=0.15)
    ax.fill_between(steps, max_values, min_values, color=color, alpha=0.15)
    # ax.axhline(n_ep, color="gray", ls="--")
    ax.set_ylim([0, 14])
    return ax


# Linear interpolation between 2 datapoints
def interpolate(pt1, pt2, x):
    x1, y1 = pt1
    x2, y2 = pt2
    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return y


def merge_by_episodes(data, min_data_axs):
    # Evaluation was done every 20 episodes
    eval_rate = 20
    x_label = np.arange(0, min_data_axs * eval_rate, eval_rate)

    # Truncate to the seed with least episodes
    data = [np.transpose(np.vstack((x_label, d[:min_data_axs, -1]))) for d in data]
    data = np.stack(data, axis=0)
    return data


def merge_by_timesteps(data, min_data_axs):
    """
    data:
        list, each element is a different seed
        data[i].shape = [n_evaluations, 2] (timestep, value)
    """
    idxs = np.arange(0, min_data_axs, 1000)
    n_pts = len(idxs)
    data_copy = []
    for d in data:
        run_values = np.zeros(shape=(n_pts, d.shape[-1]))
        interp = interp1d(d[:, 0], d[:, 1], kind="linear", fill_value="extrapolate")
        run_values[:, 1] = interp(idxs)
        run_values[:, 0] = idxs
        data_copy.append(run_values)
    # data = [d[:min_data_axs] for d in data]
    data = np.stack(data_copy, axis=0)
    return data


# Data is a list
def plot_experiments(
    data,
    show=True,
    save=True,
    n_ep=6,
    save_name="return",
    metric="return",
    save_folder="./analysis/figures/",
    x_label="timesteps",
    y_label="Completed tasks",
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), sharey=True)
    # ax.set_title("Evaluation")

    cm = plt.get_cmap("tab10")
    colors = cm(np.linspace(0, 1, len(data)))

    for exp_data, c in zip(data, colors):
        name, data = exp_data
        ax = plot_data(data, ax, n_ep=n_ep, label=name, color=c, stats_axis=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left")
    # fig.suptitle("%s" % (metric.title()))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_folder, "%s.png" % save_name), bbox_inches="tight", pad_inches=0)


def plot_by_time(plot_dict, csv_dir="./results_csv/"):
    data, labels = [], []
    for exp_name, label in plot_dict.items():
        # Skip wall time
        csv_dir = os.path.abspath(csv_dir)
        files = glob.glob("%s/*%s*success*.csv" % (csv_dir, exp_name))
        data.append(pd.read_csv(files[0]).to_numpy())
        labels.append(label)
    # search_res = re.search(r"\((.*?)\)", files[0])
    # if search_res:
    #     search_res = search_res.group(1)
    #     n_eval_ep = int(search_res[:-2])  # Remove "ep"
    # else:
    #     n_eval_ep = 10

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), sharey=True)
    cm = plt.get_cmap("cool")
    colors = cm(np.linspace(0, 1, len(data)))
    colors = [[0, 0, 1, 1], [1, 0, 0, 1]]
    smooth_window = 5
    for exp_data, c, label in zip(data, colors, labels):
        d = exp_data[:, 2]
        d = np.array(pd.Series(d).rolling(smooth_window, min_periods=smooth_window).mean())
        d[np.isnan(d)] = 0
        time = np.linspace(0, 2, num=len(d))
        ax.plot(time, d, color=c, label=label)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Success rate")
    ax.legend(loc="upper left")

    basename = os.path.basename(os.path.split(files[0])[0])
    save_folder = "./analysis/figures/"
    save_folder = os.path.join(save_folder, basename)
    save_name = "real_world_mean_success"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.show()
    fig.savefig(os.path.join(save_folder, "%s.png" % save_name), bbox_inches="tight", pad_inches=0)


# Plot validation data for a single experiment, multiple seeds
def seeds_mean(files, top_row=-1, data_merge_fnc=merge_by_timesteps):
    data = []
    for file_n in files:
        # Skip wall time
        data.append(pd.read_csv(file_n).to_numpy()[:top_row, 1:])
    search_res = re.search(r"\((.*?)\)", files[0])
    if search_res:
        search_res = search_res.group(1)
        n_eval_ep = int(search_res[:-2])  # Remove "ep"
    else:
        n_eval_ep = 10

    # Last ts logged for each run
    if data_merge_fnc == merge_by_timesteps:
        min_data_axs = min([d[-1][0] for d in data])
    else:
        min_data_axs = min([d.shape[0] for d in data])
    # Change timesteps by episodes -> x axis will show every n episodes result
    data = data_merge_fnc(data, min_data_axs)

    return data, n_eval_ep


def plot_eval_and_train(
    eval_files, train_files, task, top_row=-1, show=True, save=True, save_name="return", metric="return"
):
    eval_data, train_data = [], []
    min_val = np.inf
    for evalFile, trainFile in zip(eval_files, train_files):
        # Skip wall time
        eval_data.append(pd.read_csv(evalFile).to_numpy()[:top_row, 1:])
        stats = pd.read_csv(trainFile).to_numpy()[:, 1:]
        train_limit = top_row * len(stats) // 100
        if train_limit < min_val:
            min_val = train_limit
        train_data.append(stats[:train_limit])
    search_res = re.search(r"\((.*?)\)", eval_files[0])
    if search_res:
        search_res = search_res.group(1)
        n_eval_ep = int(search_res[:-2])  # Remove "ep"
    else:
        n_eval_ep = 10

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    train_data = [run[:min_val] for run in train_data]
    train_data = np.stack(train_data, axis=0)
    axs[0].set_title("Training")
    axs[0] = plot_data(train_data, axs[0], stats_axis=0)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel(metric.title())

    eval_data = np.stack(eval_data, axis=0)
    axs[1].set_title("Evaluation")
    axs[1] = plot_data(eval_data, axs[1], stats_axis=0)
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel("Mean %s over %s episodes" % (metric, n_eval_ep))
    fig.suptitle("%s %s" % (task.title(), metric.title()))

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")
    if save:
        fig.savefig("./results/figures/%s.png" % save_name)
    if show:
        plt.show()


def get_mean_and_std(
    exp_name="slide", metric="return", csv_folder="./results/results_csv/", data_merge_fnc=merge_by_timesteps
):
    if metric == "return":
        eval_files = glob.glob("%s*%s*eval*return*.csv" % (csv_folder, exp_name))
        # train_files = \
        #   glob.glob("%s*%s*train*return*.csv" % (csv_folder, exp_name))
    elif metric == "success":
        eval_files = glob.glob("%s*%s*eval*success*.csv" % (csv_folder, exp_name))
    else:  # episode length
        eval_files = glob.glob("%s*%s*eval*length*.csv" % (csv_folder, exp_name))
        # train_files = \
        #   glob.glob("%s*%s*train*length*.csv" % (csv_folder, exp_name))
        # metric = "episode length"
    # assert len(eval_files) == len(train_files)
    if len(eval_files) == 0:
        print("no files Match %s in %s" % (exp_name, csv_folder))
        return
    experiment_data, n_eval_ep = seeds_mean(eval_files, data_merge_fnc=data_merge_fnc)
    return experiment_data, n_eval_ep


def plot_by_timesteps(plot_dict, csv_dir="./results_csv/"):
    # metrics = ["return", "episode length"]
    metrics = ["success"]
    experiments_data = []
    for metric in metrics:
        for exp_name, title in plot_dict.items():
            mean_data, n_eval_ep = get_mean_and_std(
                exp_name, csv_folder=csv_dir, metric=metric, data_merge_fnc=merge_by_timesteps
            )
            experiments_data.append([title, mean_data])
        save_name = os.path.basename(os.path.normpath(csv_dir)) + "_%s_by_timesteps" % metric
        min_axs = min([d[-1][0].shape[0] - 1 for d in experiments_data])
        for i in range(len(experiments_data)):
            experiments_data[i][-1] = experiments_data[i][-1][:, :min_axs]
        plot_experiments(
            experiments_data,
            n_ep=n_eval_ep,
            show=True,
            save=True,
            save_name=save_name,
            save_folder="./analysis/figures/",
            metric=metric,
            x_label="timesteps",
            y_label="Completed tasks",
        )


def plot_by_episodes(plot_dict, csv_dir="./results_csv/"):
    # metrics = ["return", "episode length"]
    metrics = ["success"]
    experiments_data = []
    for metric in metrics:
        min_ep = np.inf
        for exp_name, title in plot_dict.items():
            # n_seeds, n_ep, 2
            mean_data, n_eval_ep = get_mean_and_std(
                exp_name, csv_folder=csv_dir, metric=metric, data_merge_fnc=merge_by_episodes
            )
            experiments_data.append([title, mean_data])
            if mean_data.shape[1] < min_ep:
                min_ep = mean_data.shape[1]
        # Crop to experiment with least episodes
        experiments_data = [[title, data[:, :min_ep]] for title, data in experiments_data]
        save_name = os.path.basename(os.path.normpath(csv_dir)) + "_%s_by_episodes" % metric
        plot_experiments(
            experiments_data,
            n_ep=n_eval_ep,
            show=True,
            save=True,
            save_name=save_name,
            save_folder="./analysis/figures/",
            metric=metric,
            x_label="Episodes",
            y_label="Completed tasks",
        )


if __name__ == "__main__":
    plot_dict = {"dense": "VAPO", "sparse": "local-SAC"}
    plot_by_time(plot_dict, csv_dir="./vapo/analysis/results_csv/pickup_real_world/")
