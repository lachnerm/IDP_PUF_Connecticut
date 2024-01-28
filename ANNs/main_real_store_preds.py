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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--data_file', required=True)

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

    root_folder = f"results_store_preds"
    Path(root_folder).mkdir(parents=True, exist_ok=True)

    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    size = 8000
    training_ids = list(range(4000))
    test_ids = list(range(4000, 8000))

    data_path = f"{tmp_folder}/{int(size * 0.5)}_results.json"

    log_folder = f"{root_folder}/{int(size * 0.5)}"
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    store_path = f"{root_folder}/preds/{int(size * 0.5)}"
    Path(store_path).mkdir(parents=True, exist_ok=True)

    data_module = PUFDataModule(hparams["bs"], folder, training_ids,
                                test_ids)
    data_module.setup()

    epochs = 30
    model = PUFGeneratorRealPD(hparams, challenge_bits, "store_preds",
                               data_module.denormalize, log_folder,
                               store_path=store_path)
    trainer = Trainer(gpus=1, max_epochs=epochs, logger=False)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    results = model.results

    with open(data_path, 'w') as file:
        json.dump(results, file)

    plot_generated_data(root_folder)


if __name__ == "__main__":
    main()
