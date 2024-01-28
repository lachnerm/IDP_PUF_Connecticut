import argparse
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer

from Generator_Real import PUFGeneratorRealPD
from modules.DataModule import PUFDataModule


def plot_generated_data(root_folder):
    plot_folder = f"{root_folder}/plots"
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    tmp_folder = f"{root_folder}/tmp"

    _, _, result_files = next(os.walk(tmp_folder))
    pcs = {}
    pc_means = {}
    used_training_sizes = set()
    for result_file in result_files:
        with open(f"{tmp_folder}/{result_file}", 'r') as file:
            file_data = json.load(file)
            training_size = result_file.split("_")[0]

            pcs[training_size] = file_data["PC"]
            pc_means[training_size] = np.mean(file_data["PC"])
            used_training_sizes.add(int(training_size))

    fig_scatter, ax_scatter = plt.subplots(figsize=(20, 15))
    fig_box, ax_box = plt.subplots(figsize=(20, 15))
    ticks = list(used_training_sizes)
    ticks.sort()
    axs = [ax_scatter, ax_box]
    for ax in axs:
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_xlabel("Set size", fontsize=18, labelpad=20)
        ax.set_ylabel("PC", fontsize=18, labelpad=20)
        ax.set_ylim([0, 1])
        ax.xaxis.set_ticks(ticks)

    box_tmp = []
    for training_size in ticks:
        box_fhds = pcs[str(training_size)]
        box_tmp.append(box_fhds)

        pc_mean = np.mean(pc_means[str(training_size)])
        ax_scatter.scatter(training_size, pc_mean, s=300, color="black")
    ax_box.boxplot(box_tmp, sym="+", patch_artist=True)
    # TODO: boxplot ticks
    ax_box.set_xticks(range(len(ticks)))
    ax_box.set_xticklabels(ticks)

    fig_scatter.savefig(fname=f"{plot_folder}/scatter.jpg", bbox_inches="tight",
                        pad_inches=0.3)
    fig_box.savefig(fname=f"{plot_folder}/box.jpg", bbox_inches="tight",
                    pad_inches=0.3)
    plt.close(fig_box)
    plt.close(fig_scatter)


def run_size_var_attack(hparams, challenge_bits, logger_name, root_folder,
                        folder):
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    test_sizes = range(1, 8)
    for test_crp_idx in test_sizes:
        tmp_file = f'{root_folder}/tmp/{test_crp_idx}_results.json'
        if not os.path.isfile(tmp_file):
            with open(tmp_file, 'w') as file:
                json.dump([], file)

    training_ids = list(range(1000))

    data_module = PUFDataModule(hparams["bs"], folder, training_ids, [])
    data_module.setup()
    model = PUFGeneratorRealPD(hparams, challenge_bits, logger_name,
                               data_module.denormalize, "")
    trainer = Trainer(gpus=1, max_epochs=100, logger=False)
    trainer.fit(model, datamodule=data_module)

    for test_crp_idx in test_sizes:
        data_path = f"{tmp_folder}/{test_crp_idx}_results.json"

        start = 1000 * test_crp_idx
        test_ids = list(range(start, start + 1000))

        data_module = PUFDataModule(hparams["bs"], folder, training_ids,
                                    test_ids)
        data_module.setup()
        trainer.test(model, datamodule=data_module)
        results = model.results

        with open(data_path, 'w') as file:
            json.dump(results, file)

    plot_generated_data(root_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--data_file', required=True)
    parser.add_argument('--name', default="unnamed")

    parser.add_argument('--clearml', action="store_true")
    parser.add_argument('--load', action="store_true")

    args = parser.parse_args()
    folder = args.f

    print("______________________________________________________")
    print("Starting run on data file", folder)

    hparams = {"bs": 16,
               "lr": 0.005,
               "beta1": 0.3,
               "beta2": 0.999,
               "ns": 64,
               }
    challenge_bits = 100

    root_folder = f"results_cycle"
    Path(root_folder).mkdir(parents=True, exist_ok=True)

    run_size_var_attack(hparams, challenge_bits, args.name, root_folder,
                        folder)


if __name__ == "__main__":
    main()
