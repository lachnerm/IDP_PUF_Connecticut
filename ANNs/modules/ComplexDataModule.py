import random

import numpy as np
import scipy.io as io
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class PUFDataset(Dataset):
    def __init__(self, folder, data_file, ids, use_complex=True):
        data = io.loadmat(f"/home/michael/Nonlinear_PUF/{folder}/{data_file}")

        self.challenges = data["cfgList"][0] - 1
        self.ids = ids
        self.responses = [np.array([r.real, r.imag]) for r in data["nfData"]] if use_complex else np.absolute(
            data["nfData"]) ** 2

        self.min_real = np.min(list(map(lambda r: r[0], self.responses)))
        self.max_real = np.max(list(map(lambda r: r[0], self.responses)))
        self.min_imag = np.min(list(map(lambda r: r[1], self.responses)))
        self.max_imag = np.max(list(map(lambda r: r[1], self.responses)))

        self.normalize_real = lambda response: 2 * (response - self.min_real) / (self.max_real - self.min_real) - 1
        self.normalize_imag = lambda response: 2 * (response - self.min_imag) / (self.max_imag - self.min_imag) - 1
        self.normalize = lambda response: (self.normalize_real(response[0]), self.normalize_imag(response[1]))

        self.denormalize_real = lambda response: ((response + 1) * (self.max_real - self.min_real) / 2) + self.min_real
        self.denormalize_imag = lambda response: ((response + 1) * (self.max_imag - self.min_imag) / 2) + self.min_imag

    def __len__(self):
        return len(self.ids)

    def denormalize(self, responses):
        result = torch.empty(0).type_as(responses)
        for response in responses:
            real = self.denormalize_real(response[0])
            imag = self.denormalize_imag(response[1])
            r = torch.stack((real, imag)).type_as(responses)
            result = torch.cat((result, torch.unsqueeze(r, 0)), 0)
        return result

    def __getitem__(self, idx):
        challenge = [int(bit) for bit in "{:08b}".format(self.challenges[self.ids[idx]])]
        challenge = torch.tensor(challenge, dtype=torch.float)

        response = self.responses[self.ids[idx]]
        response = torch.tensor(self.normalize(response), dtype=torch.float)
        return challenge, response


class ComplexPUFDataModule(LightningDataModule):
    def __init__(self, batch_size, folder, data_file, training_size):
        super().__init__()
        self.batch_size = batch_size
        self.folder = folder
        self.data_file = data_file
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 4, "pin_memory": True, "shuffle": True}
        self.val_test_kwargs = {"batch_size": self.batch_size, "num_workers": 4, "pin_memory": True}
        self.training_size = training_size

    def setup(self):
        training_ids = list(range(256))
        random.shuffle(training_ids)
        training_ids = training_ids[:self.training_size]

        test_ids = list(set(list(range(256))).symmetric_difference(set(training_ids)))
        random.shuffle(test_ids)
        test_ids = test_ids[:12]

        self.train_dataset = PUFDataset(self.folder, self.data_file, training_ids)
        self.test_dataset = PUFDataset(self.folder, self.data_file, test_ids)
        self.denormalize = self.test_dataset.denormalize

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)
