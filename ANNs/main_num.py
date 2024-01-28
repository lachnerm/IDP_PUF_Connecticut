import argparse
import json
import os
import random
from pathlib import Path

import itertools
import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from Generator_Num import PUFGenerator, PUFGeneratorPD
from modules.ComplexDataModule import ComplexPUFDataModule
from modules.DataModule import PUFDataModule


def plot_generated_data_new(root_folder):
    plot_folder = f"{root_folder}/plots"
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    tmp_folder = f"{root_folder}/tmp"

    _, _, result_files = next(os.walk(tmp_folder))
    pcs = {}
    pc_means = {}
    used_training_sizes = set()
    for result_file in result_files:
        with open(f"{tmp_folder}/{result_file}", 'r') as file:
            pass


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
    ax_box.set_xticks(range(4))
    ax_box.set_xticklabels(ticks)

    fig_scatter.savefig(fname=f"{plot_folder}/scatter.jpg", bbox_inches="tight", pad_inches=0.3)
    fig_box.savefig(fname=f"{plot_folder}/box.jpg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig_box)
    plt.close(fig_scatter)


def run_size_var_attack(hparams, challenge_bits, logger_name, root_folder, folder, use_complex, pd):
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    if folder == "Disordered":
        size = 2048
    elif folder == "Disordered_12bit":
        size = 4096
    else:
        size = 256

    training_sizes = [int(pctg * size) for pctg in [0.5, 0.7, 0.9, 0.95]]

    test_size = size - int(0.95 * size)
    test_ids = list(range(size))
    random.shuffle(test_ids)
    test_ids = test_ids[:test_size]

    # test_ids = [16, 131, 46, 94, 108, 166, 167, 119, 242, 142, 195, 161, 197]

    for training_size in training_sizes:
        tmp_file = f'{root_folder}/tmp/{training_size}_results.json'
        if not os.path.isfile(tmp_file):
            with open(tmp_file, 'w') as file:
                json.dump([], file)

    for training_size in training_sizes:
        training_ids = list(set(list(range(size))).symmetric_difference(set(test_ids)))[:training_size]

        data_path = f"{tmp_folder}/{training_size}_results.json"
        log_folder = f"{root_folder}/{training_size}"
        Path(log_folder).mkdir(parents=True, exist_ok=True)

        if use_complex:
            data_module = ComplexPUFDataModule(hparams["bs"], folder, training_ids, test_ids)
        else:
            data_module = PUFDataModule(hparams["bs"], folder, training_ids, test_ids)
        data_module.setup()

        if pd:
            model = PUFGeneratorPD(hparams, challenge_bits, logger_name, data_module.denormalize, log_folder,
                                   use_complex=use_complex)
            trainer = Trainer(gpus=1, max_epochs=300, logger=False)
        else:
            model = PUFGenerator(hparams, challenge_bits, logger_name, data_module.denormalize, log_folder,
                                 use_complex=use_complex)
            logger = TensorBoardLogger(f'runs',
                                       name=f"{folder}_{training_size}{f'_{logger_name}' if logger_name != 'unnamed' else ''}")
            trainer = Trainer(gpus=1, max_epochs=300, logger=logger)
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        results = model.results

        with open(data_path, 'w') as file:
            json.dump(results, file)

    plot_generated_data(root_folder)


def run_regular_attack(hparams, challenge_bits, logger_name, root_folder, folder, use_complex, pd):
    if folder == "Disordered":
        size = 2048
    else:
        size = 256
    training_size = int(size * 0.95)
    test_size = size - training_size
    samples = list(range(size))
    random.shuffle(samples)
    training_ids = samples[:training_size]
    test_ids = list(set(list(range(size))).symmetric_difference(set(training_ids)))[:test_size]

    if use_complex:
        data_module = ComplexPUFDataModule(hparams["bs"], folder, training_ids, test_ids)
    else:
        data_module = PUFDataModule(hparams["bs"], folder, training_ids, test_ids)
    data_module.setup()

    if pd:
        model = PUFGeneratorPD(hparams, challenge_bits, logger_name, data_module.denormalize, root_folder,
                               use_complex=use_complex)
        trainer = Trainer(gpus=1, max_epochs=300, logger=False)
    else:
        model = PUFGenerator(hparams, challenge_bits, logger_name, data_module.denormalize, root_folder,
                             use_complex=use_complex)
        logger = TensorBoardLogger(f'runs', name=f"{folder}_{logger_name}")
        trainer = Trainer(gpus=1, max_epochs=300, logger=logger)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="unnamed")
    parser.add_argument('--sv', '--size_var', action="store_true")
    parser.add_argument('--c', '--complex', action="store_true")

    parser.add_argument('--clearml', action="store_true")
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--pd', action="store_true")
    parser.add_argument('--f', '--data_file', required=True)
    parser.add_argument('--r', '--runs', type=int, default=0)

    args = parser.parse_args()

    folder = args.f
    if "Disordered" in folder:
        challenge_bits = 12
        hparams = {"bs": 32,
                   "lr": 0.0005,
                   "beta1": 0.3,
                   "beta2": 0.999,
                   "ns": 64,
                   "act": "GELU"
                   }
    else:
        challenge_bits = 8
        hparams = {"bs": 16,
                   "lr": 0.005,
                   "beta1": 0.5,
                   "beta2": 0.999,
                   "ns": 96,
                   }

    '''if args.clearml:
        task = Task.init(project_name="Nonlinear PUF 2048", task_name="PUF 2048", task_type="training")
        task.connect(hparams)'''

    root_folder = f"results{'_complex' if args.c else ''}{'_size_var' if args.sv else ''}/{folder}"
    Path(root_folder).mkdir(parents=True, exist_ok=True)

    params = [hparams, challenge_bits, args.name, root_folder, folder, args.c, args.pd]

    if args.sv:
        run_size_var_attack(*params)
    else:
        run_regular_attack(*params)


if __name__ == "__main__":
    main()
