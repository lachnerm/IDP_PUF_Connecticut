import argparse
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from Generator_Real import PUFGenerator, PUFGeneratorRealPD
from modules.DataModule import PUFDataModule
from test_ids import test_ids_1k, test_ids_8k


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
                        folder, pd):
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    size = 8000 if "8k" in folder else 1000
    training_sizes = [int(pctg * size) for pctg in [0.5, 0.7, 0.9]]

    if folder == "Real" or "circle" in folder:
        test_ids = test_ids_1k
    elif "8k" in folder:
        test_ids = test_ids_8k
    else:
        print("Unknown folder", folder)
        exit()

    for training_size in training_sizes:
        tmp_file = f'{root_folder}/tmp/{training_size}_results.json'
        if not os.path.isfile(tmp_file):
            with open(tmp_file, 'w') as file:
                json.dump([], file)

    for training_size in training_sizes:
        training_ids = list(
            set(list(range(size))).symmetric_difference(set(test_ids)))[
                       :training_size]

        '''if "circle" in folder:
            test_ids = [x + 7000 for x in test_ids]
            training_ids = [x + 7000 for x in training_ids]'''

        data_path = f"{tmp_folder}/{training_size}_results.json"
        log_folder = f"{root_folder}/{training_size}"
        Path(log_folder).mkdir(parents=True, exist_ok=True)

        data_module = PUFDataModule(hparams["bs"], folder, training_ids,
                                    test_ids)
        data_module.setup()

        if pd:
            epochs = 100 if not "8k" in folder else 30
            model = PUFGeneratorRealPD(hparams, challenge_bits, logger_name,
                                       data_module.denormalize, log_folder)
            trainer = Trainer(gpus=1, max_epochs=epochs, logger=False)
        else:
            model = PUFGenerator(hparams, challenge_bits, logger_name,
                                 data_module.denormalize, log_folder)
            logger_name = f"{folder}_{training_size}{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
            logger = TensorBoardLogger(f'runs', name=logger_name)
            trainer = Trainer(gpus=1, max_epochs=75, logger=logger)
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        results = model.results

        with open(data_path, 'w') as file:
            json.dump(results, file)

    plot_generated_data(root_folder)


def run_regular_attack(hparams, challenge_bits, logger_name, root_folder,
                       folder, pd):
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    size = 8000 if "8k" in folder else 1000
    training_size = int(size * 0.9)

    data_path = f"{tmp_folder}/results_{training_size}.json"

    if folder == "Real" or "circle" in folder:
        test_ids = test_ids_1k
    elif "8k" in folder:
        test_ids = test_ids_8k
    else:
        print("Unknown folder", folder)
        exit()

    training_ids = list(
        set(list(range(size))).symmetric_difference(set(test_ids)))[
                   :training_size]

    data_module = PUFDataModule(hparams["bs"], folder, training_ids, test_ids)
    data_module.setup()

    model = PUFGenerator(hparams, challenge_bits, logger_name,
                         data_module.denormalize, root_folder)
    logger = TensorBoardLogger(f'runs',
                               name=f"{folder}_single_run{f'_{logger_name}' if logger_name != 'unnamed' else ''}")
    trainer = Trainer(gpus=1, max_epochs=50, logger=logger)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    results = model.results
    print("Mean PC:", np.mean(results["PC"]))
    print(results)
    print("Mean PC:", np.mean(results["PC"]))

    with open(data_path, 'w') as file:
        json.dump(results, file)

    plot_generated_data(root_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--data_file', required=True)
    parser.add_argument('--name', default="unnamed")
    parser.add_argument('--sv', '--size_var', action="store_true")

    parser.add_argument('--clearml', action="store_true")
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--pd', action="store_true")

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

    '''if args.clearml:
        task = Task.init(project_name="Nonlinear PUF 2048", task_name="PUF 2048", task_type="training")
        task.connect(hparams)'''

    root_folder = f"results{'_size_var' if args.sv else ''}/{folder}"
    Path(root_folder).mkdir(parents=True, exist_ok=True)

    print(
        f"Running WITH{'OUT' if not args.sv else ''} size var in mode {'PROD' if args.pd else 'DEV'}")

    if args.sv:
        run_size_var_attack(hparams, challenge_bits, args.name, root_folder,
                            folder, args.pd)
    else:
        run_regular_attack(hparams, challenge_bits, args.name, root_folder,
                           folder, args.pd)


if __name__ == "__main__":
    main()
