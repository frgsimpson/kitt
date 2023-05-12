"""
Load a resnet or resnext model.
 See https://github.com/osmr/imgclsmob/tree/master/keras_
"""

import numpy as np

from kitt.prototype.classifier.resnet import resnet12
from kitt.prototype.classifier.resnext import *

_models = {
    "resnext14_16x4d": resnext14_16x4d,
    "resnext14_32x2d": resnext14_32x2d,
    "resnext14_32x4d": resnext14_32x4d,
    "resnext26_16x4d": resnext26_16x4d,
    "resnext26_32x2d": resnext26_32x2d,
    "resnext26_32x4d": resnext26_32x4d,
    "resnext38_32x4d": resnext38_32x4d,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x4d": resnext101_32x4d,
    "resnext101_64x4d": resnext101_64x4d,
    "resnet12": resnet12,
}


def get_model(name, **kwargs):
    """
    Get supported model.
    Parameters:
    ----------
    name : str
        Name of model.
    Returns
    -------
    Model
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net


if __name__ == "__main__":
    # smoke test - and demo of typical usage
    N_CHANNELS = 1
    network = "resnet12"  # "resnext14_16x4d"

    resolution = 28  # resnet default 224
    image_shape = (resolution, resolution)

    net = get_model(
        network, pretrained=False, in_channels=N_CHANNELS, in_size=image_shape, classes=10
    )
    # can also take width_scale and conv1_stride args

    x = np.zeros((1, resolution, resolution, N_CHANNELS), np.float32)
    y = net.predict(x)
    print(y)
