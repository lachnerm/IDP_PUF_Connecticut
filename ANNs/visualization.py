import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_results(result_folder, folders, pctgs, sizes, metrics):
    results = {dataset: {size: {metric: [] for metric in metrics} for size in pctgs} for dataset in folders}

    for folder, size in zip(folders, sizes):
        _, _, data_files = next(os.walk(f"{result_folder}/{folder}/tmp"))
        for data_file in data_files:
            with open(f"{result_folder}/{folder}/tmp/{data_file}", "r") as file:
                data = json.load(file)
                t_size = int(data_file.split("_")[0])
                pctg = f"{round((t_size / size) * 100)}%"

                results[folder][pctg]["PC"] += data["PC"]
                results[folder][pctg]["RE"] += data["Rel_Err"]

    return results


def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols}) \
                   .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols}) \
                   .append(df.loc[lens == 0, idx_cols]).fillna(fill_value) \
                   .loc[:, df.columns]


def get_other_results():
    result_folder = "results_size_var"
    folders = ["Disordered", "SmoothSurface", "ScatteringSurface"]
    sizes = [2048, 256, 256]
    pctgs = ["50%", "70%", "90%", "95%"]
    metrics = ["PC", "RE"]
    return build_results(result_folder, folders, pctgs, sizes, metrics)


def get_volume_scatterer_results():
    result_folder = "results_size_var"
    folders = ["VolumeScatterers400", "VolumeScatterers800", "VolumeScatterers1000"]
    sizes = [256, 256, 256]
    pctgs = ["50%", "70%", "90%", "95%"]
    metrics = ["PC", "RE"]
    return build_results(result_folder, folders, pctgs, sizes, metrics)


def get_disordered_results():
    result_folder = "results_size_var"
    folders = ["Disordered", "Disordered_12bit"]
    sizes = [2048, 4096]
    pctgs = ["50%", "70%", "90%", "95%"]
    metrics = ["PC", "RE"]
    return build_results(result_folder, folders, pctgs, sizes, metrics)


def get_real_results():
    result_folder = "results_size_var"
    folders = ["Real"]
    sizes = [1000]
    pctgs = ["10%", "30%", "50%", "70%", "90%"]
    metrics = ["PC", "RE"]
    return build_results(result_folder, folders, pctgs, sizes, metrics)


def main():
    # all_results = [get_other_results(), get_volume_scatterer_results()]
    #all_results = [get_disordered_results()]
    all_results = [get_real_results()]
    plot_paths = ["results_plot.png"]

    for results, plot_path in zip(all_results, plot_paths):
        results_pandas = {
            (data_type, pctg, idx): (results[data_type][pctg]["PC"][idx], results[data_type][pctg]["RE"][idx])
            for data_type in results.keys()
            for pctg in results[data_type].keys()
            for idx in range(len(results[data_type][pctg]["PC"]))
        }
        df = pd.DataFrame.from_dict(results_pandas, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=("Type", "Size", "CRP"))
        df.columns = ["Pearson correlation", "Relative error"]

        df = df.stack().reset_index()
        df.columns = ["Dataset", "Training Size", "CRP", "Metric", "Value"]

        plt.rc("axes", labelsize=15)
        plt.rc("axes", titlesize=15)

        g = sns.catplot(x="Training Size", y="Value", row="Metric", col="Dataset", data=df, kind="box", sharey="row",
                        legend=False, margin_titles=True, palette=["gold", "darkorange", "indianred", "darkred"])
        '''g = sns.catplot(x="Training Size", y="Value", row="Metric", col="Dataset", data=df, kind="swarm", sharey="row",
                        legend=False, margin_titles=True, palette=["gold", "darkorange", "indianred", "darkred"])
        '''
        g.fig.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        axes = g.axes
        '''for axs in axes[0]:
            for step in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
            axs.set_ylim(0, 1)'''
        for axs in axes[1]:
            '''for step in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)'''
            axs.set_ylim(0, 1.2)

        for axs in axes.flatten():
            axs.tick_params(axis="both", labelsize=12)
            for step in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        plt.savefig(fname=plot_path, bbox_inches="tight", pad_inches=0.1)

    plot_paths = ["results_plot_scatter.png"]
    for results, plot_path in zip(all_results, plot_paths):
        results_pandas = {
            (data_type, pctg): (np.mean(results[data_type][pctg]["PC"]), np.mean(results[data_type][pctg]["RE"]))
            for data_type in results.keys()
            for pctg in results[data_type].keys()
        }
        df = pd.DataFrame.from_dict(results_pandas, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=("Type", "Size"))
        df.columns = ["Pearson correlation", "Relative error"]

        df = df.stack().reset_index()
        df.columns = ["Dataset", "Training Size", "Metric", "Value"]

        plt.rc("axes", labelsize=15)
        plt.rc("axes", titlesize=15)

        g = sns.catplot(x="Training Size", y="Value", row="Metric", col="Dataset", data=df, kind="swarm",
                        legend=False, margin_titles=True, palette=["gold", "darkorange", "indianred", "darkred"], s=10)
        g.fig.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        axes = g.axes
        for axs in axes.flatten():
            axs.tick_params(axis="both", labelsize=12)
            for step in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        plt.savefig(fname=plot_path, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    main()
