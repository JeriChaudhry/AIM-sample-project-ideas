import gzip
import numpy as np
import torch
from scipy.ndimage import rotate


def extract_binary_dataset(filepath, size_less_imgs):
    step = 0
    pixels = []
    with gzip.open(filepath, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        size = int.from_bytes(f.read(4), byteorder="big")
        nrows = int.from_bytes(f.read(4), byteorder="big")
        ncols = int.from_bytes(f.read(4), byteorder="big")
        print(f"Magic no.: {magic}\nNum images: {size}\nImage res: {nrows}x{ncols}")
        total = size_less_imgs * nrows * ncols
        d = f.read(1)
        while step < size_less_imgs * 28 * 28:
            num = int.from_bytes(d, byteorder="big")
            pixels.append(num)
            d = f.read(1)
            print(f"{step}/{total} - {100*step/total:.3f}%", end="\r")
            step += 1
    print("")

    print("Num Pixels: " + str(len(pixels)))

    # Import, Unzip, & read Train Images! (Part 2)

    size_less_pixels = size_less_imgs * ncols * nrows
    data = pixels[:size_less_pixels]
    # data = []
    # step = 0
    # total = len(pixels)
    # for p in pixels:
    #     data.append(int.from_bytes(p, byteorder="big"))
    #     print(f"{step}/{total}", end="\r")
    #     step += 1

    # print("")
    data = np.array(data, dtype=np.uint8).reshape((size_less_imgs, nrows, ncols))

    data_list = []

    for a in data:
        a_fixed = np.fliplr(rotate(a, 270))
        data_list.append(a_fixed)

    data_list = np.array(data_list).reshape(size_less_imgs, nrows, ncols)
    data_list = torch.from_numpy(data_list)

    return data_list


def extract_labels(filepath, size_less_imgs):

    Labels = []
    with gzip.open(filepath, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        size = int.from_bytes(f.read(4), byteorder="big")

        print(f"Magic no.: {magic}\nNum items: {size}")
        d = f.read(1)
        while d != b"":
            Labels.append(d)
            d = f.read(1)

    Labels = Labels[0:size_less_imgs]
    Labels_data = []
    step = 0
    total = len(Labels)
    for p in Labels:
        Labels_data.append(int.from_bytes(p, byteorder="big"))
        print(f"{step}/{total}", end="\r")
        step += 1
    Labels_data = torch.ByteTensor(Labels_data)
    # Labels_data = np.array(Labels_data, dtype=np.uint8)

    return Labels_data
