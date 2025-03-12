import numpy as np
from numpy.typing import NDArray
from pathlib import Path


def get_mnist() -> tuple[NDArray, NDArray]:

    path = Path(__file__).parent.parent / 'data' / 'mnist.npz'

    with np.load(path) as f:
        images, labels = f["x_train"], f["y_train"]
        
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels
