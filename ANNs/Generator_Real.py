import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

import board
import utils
from modules.GeneratorReal import GeneratorReal


class PUFGenerator(LightningModule):
    def __init__(self, hparams, challenge_bits, name, denormalize, log_folder):
        super().__init__()
        self.hparams.update(hparams)
        self.name = name
        self.denormalize = denormalize
        self.log_folder = log_folder

        self.challenge_bits = challenge_bits
        self.train_log = {}
        self.val_log = {}
        self.test_log = {}
        self.challenge_shape = (10, 10)
        self.response_shape = (512, 512)
        self.generator = GeneratorReal(self.hparams.ns, challenge_bits, 10)

        self.generator.apply(utils.weights_init)

    def loss_function(self, real_response, gen_response):
        criterion = nn.MSELoss()
        l1_loss = criterion(real_response, gen_response)
        return l1_loss

    def on_train_epoch_start(self):
        self.grad_fig, self.gen_grad = board.create_gradient_figure("Generator")

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)
        loss = self.loss_function(real_response, gen_response)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        if batch_idx == 0:
            self.train_log["challenge"] = challenge.detach()
            self.train_log["real_response"] = real_response.detach()
            self.train_log["gen_response"] = gen_response.detach()

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        gen_avg_loss = torch.stack(
            [output["loss"] for output in outputs]).mean()

        self.log_gen_output_figure(self.train_log, "Train")

        self.logger.experiment.add_figure("Gradients", self.grad_fig,
                                          self.current_epoch)
        self.logger.experiment.add_scalars("Training Loss",
                                           {"Generator Loss": gen_avg_loss},
                                           self.current_epoch)

    def validation_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        if batch_idx == 0:
            self.val_log["challenge"] = challenge.detach()
            self.val_log["real_response"] = real_response.detach()
            self.val_log["gen_response"] = gen_response.detach()

        diff_map_mean = utils.calc_diff_map_mean(real_response, gen_response)
        rel_error_mean = (
                np.linalg.norm(
                    real_response.cpu() - gen_response.cpu()) / np.linalg.norm(
            real_response.cpu()))

        pc = utils.calc_pear_coeff(real_response, gen_response).item()
        return {'diff_map_mean': diff_map_mean, 'pc': pc,
                'rel_err': rel_error_mean}

    def test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        if batch_idx == 0:
            self.test_log["challenge"] = challenge
            self.test_log["real_response"] = real_response
            self.test_log["gen_response"] = gen_response

        rel_error = utils.calc_rel_error_tensor(real_response, gen_response)
        pc = utils.calc_pc_tensor(real_response, gen_response)
        return {'pc': pc, 'rel_err': rel_error}

    def validation_epoch_end(self, outputs):
        epoch = self.current_epoch

        diff_map_mean = np.mean([output["diff_map_mean"] for output in outputs])
        pc = np.mean([output["pc"] for output in outputs])
        rel_err = np.mean([output["rel_err"] for output in outputs])

        # optuna
        self.log("val_pc", pc)
        self.log("hp_metric", pc, on_step=False, on_epoch=True)

        self.logger.experiment.add_scalar(
            f"Validation Average Difference Map Mean", diff_map_mean, epoch)
        self.logger.experiment.add_scalar(f"Validation Average Relative Error",
                                          rel_err, epoch)
        self.logger.experiment.add_scalar(
            f"Validation Average Pearson Correlation", pc, epoch)
        self.log_gen_output_figure(self.val_log, "Validation")

    def test_epoch_end(self, outputs):
        pc = list(
            np.concatenate([output["pc"] for output in outputs], axis=0).astype(
                np.float))
        rel_err = list(np.concatenate([output["rel_err"] for output in outputs],
                                      axis=0).astype(np.float))
        self.logger.experiment.add_scalar(f"Test Average Relative Error",
                                          np.mean(rel_err))
        self.logger.experiment.add_scalar(f"Test Average Pearson Correlation",
                                          np.mean(pc))

        gen_output_figure = self.log_gen_output_figure(self.test_log, "Test")
        gen_output_figure.savefig(
            fname=f"{self.log_folder}/{self.name}_Test.jpg",
            bbox_inches='tight', pad_inches=0)
        self.results = {"PC": pc, "Rel_Err": rel_err}

    def backward(self, trainer, loss, optimizer_idx):
        super().backward(trainer, loss, optimizer_idx)
        if self.trainer.global_step % 3 == 0:
            board.plot_grad_flow(self.generator.named_parameters(),
                                 self.gen_grad)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(),
                                     self.hparams.lr,
                                     (self.hparams.beta1, self.hparams.beta2))
        return optimizer

    # Generator output figure should only be called during Validation or Testing
    def log_gen_output_figure(self, log, log_str):
        gen_output_figure = board.get_gen_output_figure_2d(log["challenge"],
                                                           log["real_response"],
                                                           log["gen_response"],
                                                           self.challenge_shape,
                                                           self.response_shape)
        self.logger.experiment.add_figure(
            f"{log_str} Real vs. Generated Output", gen_output_figure,
            self.current_epoch)
        return gen_output_figure


class PUFGeneratorRealPD(LightningModule):
    def __init__(self, hparams, challenge_bits, name, denormalize, log_folder,
                 store_path=""):
        super().__init__()
        self.hparams.update(hparams)
        self.name = name
        self.denormalize = denormalize
        self.log_folder = log_folder
        self.store_path = store_path

        self.generator = GeneratorReal(self.hparams.ns, challenge_bits, 10)

        self.generator.apply(utils.weights_init)

        self.challenges = []
        self.preds = []

    def loss_function(self, real_response, gen_response):
        criterion = nn.MSELoss()
        l1_loss = criterion(real_response, gen_response)
        return l1_loss

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)
        loss = self.loss_function(real_response, gen_response)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        if self.store_path:
            self.challenges.append(challenge.int())
            self.preds.append(gen_response.squeeze())

        rel_error = utils.calc_rel_error_tensor(real_response, gen_response)
        pc = utils.calc_pc_tensor(real_response, gen_response)
        return {'pc': pc, 'rel_err': rel_error}

    def test_epoch_end(self, outputs):
        pc = list(
            np.concatenate([output["pc"] for output in outputs], axis=0).astype(
                np.float))
        rel_err = list(np.concatenate([output["rel_err"] for output in outputs],
                                      axis=0).astype(np.float))
        utils.store_preds(torch.cat(self.challenges), torch.cat(self.preds),
                          self.store_path)
        self.results = {"PC": pc, "Rel_Err": rel_err}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(),
                                     self.hparams.lr,
                                     (self.hparams.beta1, self.hparams.beta2))
        return optimizer


class PUFGeneratorOptuna(LightningModule):
    def __init__(self, hparams, challenge_bits, denormalize):
        super().__init__()
        self.hparams.update(hparams)
        self.denormalize = denormalize
        self.generator = GeneratorReal(self.hparams.ns, challenge_bits, 10)
        self.generator.apply(utils.weights_init)

    def loss_function(self, real_response, gen_response):
        criterion = nn.MSELoss()
        l1_loss = criterion(real_response, gen_response)
        return l1_loss

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)
        loss = self.loss_function(real_response, gen_response)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        pc = utils.calc_pear_coeff(real_response, gen_response).item()
        return {'pc': pc}

    def validation_epoch_end(self, outputs):
        pc = np.mean([output["pc"] for output in outputs])
        # optuna
        self.log("val_pc", pc)
        self.log("hp_metric", pc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(),
                                     self.hparams.lr,
                                     (self.hparams.beta1, self.hparams.beta2))
        return optimizer
