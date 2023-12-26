from torch import nn

from cplxtorch import nn as cnn
import unetv1 as _unetv1
import unetv1_real as _unetv1_real


def aft_L11bL11b(features: int, **kwargs):
    return nn.Sequential(
        cnn.Linear(features, features),
        cnn.Linear(features, features)
    )


def aft_NL11bNL11b(features: int, **kwargs):
    return nn.Sequential(
        cnn.NaiveLinear(features, features),
        cnn.NaiveLinear(features, features)
    )


def aft_L11ubL11ub(features: int, **kwargs):
    return nn.Sequential(
        cnn.Linear(features, features, bias=False),
        cnn.Linear(features, features, bias=False)
    )


def aft_L12ubRL21ub(features: int, **kwargs):
    return nn.Sequential(
        cnn.Linear(features, features * 2, bias=False),
        cnn.ReLU(),
        cnn.Linear(features * 2, features, bias=False)
    )


def aft_L12ubRL21ub(features: int, **kwargs):
    return nn.Sequential(
        cnn.Linear(features, features * 2, bias=False),
        cnn.ReLU(),
        cnn.Linear(features * 2, features, bias=False)
    )


def aft_L12ubRL22ubRL21ub(features: int, **kwargs):
    return nn.Sequential(
        cnn.Linear(features, features * 2, bias=False),
        cnn.ReLU(),
        cnn.Linear(features * 2, features * 2, bias=False),
        cnn.ReLU(),
        cnn.Linear(features * 2, features, bias=False)
    )


def aft_L12bRL22bRL21b(features: int, **kwargs):
    # 0.18564329986572264 ± 0.008576731411242733 at epoch 100
    return nn.Sequential(
        cnn.Linear(features, features * 2),
        cnn.ReLU(),
        cnn.Linear(features * 2, features * 2),
        cnn.ReLU(),
        cnn.Linear(features * 2, features)
    )


def aft_L12bLRL22bLRL21b(features: int, negative_slope=.1, **kwargs):
    # 0.00028914167685143186 ± 0.00013909085978639266 at epoch 100
    return nn.Sequential(
        cnn.Linear(features, features * 2),
        cnn.LeakyReLU(negative_slope=negative_slope),
        cnn.Linear(features * 2, features * 2),
        cnn.LeakyReLU(negative_slope=negative_slope),
        cnn.Linear(features * 2, features)
    )


def aft_L12ubLRL22ubLRL21ub(features: int, negative_slope=.1, **kwargs):
    # 0.10637775365114212 ± 0.002910950646986401 at epoch 100
    return nn.Sequential(
        cnn.Linear(features, features * 2, bias=False),
        cnn.LeakyReLU(negative_slope=negative_slope),
        cnn.Linear(features * 2, features * 2, bias=False),
        cnn.LeakyReLU(negative_slope=negative_slope),
        cnn.Linear(features * 2, features, bias=False)
    )


def aft_L14bLRL44bLRL41b(features: int, negative_slope=.1, **kwargs):
    # 0.0003471386195858941 ± 9.271368396639632e-05 at epoch 100
    return nn.Sequential(
        cnn.Linear(features, features * 4),
        cnn.LeakyReLU(negative_slope=negative_slope),
        cnn.Linear(features * 4, features * 4),
        cnn.LeakyReLU(negative_slope=negative_slope),
        cnn.Linear(features * 4, features)
    )


def unetv1(
    in_channels: int,
    out_channels: int,
    layer_channels: list[int],
    attention: bool = True,
    norm_type='GroupNorm',
    **kwargs
):
    _unetv1.norm_type = norm_type
    return _unetv1.UNet(in_channels, out_channels, layer_channels, attention)


def unetv1_real(
    in_channels: int,
    out_channels: int,
    layer_channels: list[int],
    attention: bool = True,
    norm_type='GroupNorm',
    **kwargs
):
    _unetv1_real.norm_type = norm_type
    return _unetv1_real.UNet(in_channels, out_channels, layer_channels, attention)
