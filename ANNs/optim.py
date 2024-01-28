import os
import random

import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

from Generator_Real import PUFGeneratorOptuna
from modules.DataModule import PUFDataModule


def objective(trial):
    folder = "Real"
    challenge_bits = 100

    size = 8000 if "8k" in folder else 1000
    training_size = int(size * 0.90)
    test_size = size - training_size

    samples = list(range(size))
    random.shuffle(samples)
    training_ids = samples[:training_size]
    test_ids = list(set(list(range(size))).symmetric_difference(set(training_ids)))[:test_size]

    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    '''checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join("./models", "trial_{}".format(trial.number)), monitor="val_pc"
    )'''

    bs = trial.suggest_categorical("bs", [8, 16])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.3, 0.9, step=0.1)
    beta2 = trial.suggest_float("beta2", 0.3, 0.9, step=0.1)
    ns = trial.suggest_int("ns", 16, 64, 8)

    hparams = {"bs": bs,
               "lr": lr,
               "beta1": beta1,
               "beta2": beta2,
               "ns": ns
               }
    hyperparameters = dict(lr=lr, beta1=beta1, beta2=beta2, ns=ns)

    data_module = PUFDataModule(hparams["bs"], folder, training_ids, test_ids)
    data_module.setup()
    model = PUFGeneratorOptuna(hparams, challenge_bits, data_module.denormalize)

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=75,
        gpus=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_pc")],
        progress_bar_refresh_rate=0
    )

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val_pc"].item()


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=43000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
