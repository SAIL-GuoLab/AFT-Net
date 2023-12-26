# Artificial Fourier Transform (AFT)-Net

[arxiv](https://arxiv.org/abs/2312.10892)

Major Contributors:

- Yanting Yang (Email: <yy3189@columbia.edu>)
- Jia Guo (Email: <jg3400@columbia.edu>)

## Start

The code is developed under `torch==2.1.2` and `python==3.10.13`. We do not guarantee the compatibility under other environment setup.

## Usage

### AFT2d/NaiveAFT2d

```python
class AFT2d(Module):
    def __init__(self, shape_in: tuple[int, int]) -> None:
        super().__init__()
        self.linear1 = torch.hub.load(
            'yangyanting233/AFT-Net',
            'aft_L11bL11b',
            features=shape_in[1]
        )
        self.linear2 = torch.hub.load(
            'yangyanting233/AFT-Net',
            'aft_L11bL11b',
            features=shape_in[0]
        )

    def forward(self, t: Tensor) -> Tensor:
        new_t = self.linear1(t)
        return new_t, self.linear2(new_t.mT).mT
```

```python
class NaiveAFT2d(Module):
    def __init__(self, shape_in: tuple[int, int]) -> None:
        super().__init__()
        self.linear1 = torch.hub.load(
            'yangyanting233/AFT-Net',
            'aft_NL11bNL11b',
            features=shape_in[1]
        )
        self.linear2 = torch.hub.load(
            'yangyanting233/AFT-Net',
            'aft_NL11bNL11b',
            features=shape_in[0]
        )

    def forward(self, t: Tensor) -> Tensor:
        new_t = self.linear1(t)
        return new_t, self.linear2(new_t.mT).mT
```

### RACUNet/RAUNet

```python
class RACUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.acunet = torch.hub.load(
            'yangyanting233/AFT-Net',
            'unetv1',
            in_channels=4,
            out_channels=4,
            layer_channels=[32,64,128,256,512],
            attention=True,
            norm_type='GroupNorm'
        )

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.acunet(input)
        return t + identity
```

```python
class RAUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.aunet = torch.hub.load(
            'yangyanting233/AFT-Net',
            'unetv1_real',
            in_channels=4,
            out_channels=4,
            layer_channels=[64, 128, 256, 512, 1024],
            attention=True,
            norm_type='GroupNorm'
        )

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.aunet(input)
        return t + identity
```

## Citation

```text
@misc{yang2023deep,
      title={Deep Learning-based MRI Reconstruction with Artificial Fourier Transform (AFT)-Net}, 
      author={Yanting Yang and Jeffery Siyuan Tian and Matthieu Dagommer and Jia Guo},
      year={2023},
      eprint={2312.10892},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
