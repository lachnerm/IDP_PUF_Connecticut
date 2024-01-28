import h5py
import mat73
import scipy.io
from PIL import Image
path = "../../data/Real_8k/data.h5"

with h5py.File(path, "r") as data:
    print(data["challenges"].shape)
    print(data["responses"].shape)
    exit()