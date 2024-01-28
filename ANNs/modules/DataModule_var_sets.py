import h5py

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class CenterCrop512(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.size = 512

    def forward(self, img):
        return center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size=512)'


def center_crop(img, crop_size=512):
    """
    Crops the center part of given size of an image (2d numpy array). The default size is 200x200 pixels, which is the
    size used for the Generator DL attack.

    :param img: image to be cropped (2d numpy array)
    :param crop_size: size of the center square that will be cropped from the image
    :return: cropped center part of the image
    """
    y_size, x_size = img.shape[-2:]
    x_start = x_size // 2 - (crop_size // 2)
    y_start = y_size // 2 - (crop_size // 2)
    if len(img.shape) == 2:
        return img[y_start:y_start + crop_size, x_start:x_start + crop_size]
    else:
        return img[:, y_start:y_start + crop_size, x_start:x_start + crop_size]


class PUFDataset(Dataset):
    def __init__(self, folder, ids, transform):
        self.data_file = f"../data/{folder}/data.h5"
        self._h5_gen = None

        with h5py.File(self.data_file, 'r') as data:
            min = data.get("min")[()]
            max = data.get("max")[()]
        self.normalize = lambda response: 2 * (response - min) / (max - min) - 1
        self.denormalize = lambda response: ((response + 1) * (
                    max - min) / 2) + min

        self.folder = folder
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator(self.data_file)
            next(self._h5_gen)

        challenge, response = self._h5_gen.send(self.ids[idx])

        response = self.transform(
            response).float() if self.transform is not None else response.float()
        challenge = torch.tensor(challenge, dtype=torch.float)
        response = self.normalize(response)
        return challenge, response

    def _get_generator(self, path):
        with h5py.File(path, 'r') as data:
            index = yield
            while True:
                c = data.get("challenges")[index]
                r = data.get("responses")[index]
                index = yield c, r


class PUFDataModule(LightningDataModule):
    def __init__(self, batch_size, folder, training_ids, test_ids):
        super().__init__()
        self.batch_size = batch_size
        self.folder1 = "Real_8k2"
        self.folder2 = "Real_8k3"
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 8,
                             "pin_memory": True, "shuffle": True}
        self.val_test_kwargs = {"batch_size": self.batch_size, "num_workers": 8,
                                "pin_memory": True, "shuffle": True}
        self.training_ids = training_ids
        self.test_ids = test_ids

    def setup(self):
        transform = transforms.Compose([
            CenterCrop512(),
            transforms.ToTensor()
        ]) if "Real" in self.folder1 else None
        self.train_dataset = PUFDataset(self.folder1, self.training_ids,
                                        transform)
        self.test_dataset = PUFDataset(self.folder2, self.test_ids, transform)

        self.denormalize = self.train_dataset.denormalize

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)
