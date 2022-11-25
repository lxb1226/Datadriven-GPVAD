import os
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torchaudio
import torchaudio.transforms as audio_transforms
import torch.nn.functional as F
torchaudio.set_audio_backend('sox_io')


def crnn10(inputdim=64, outputdim=527, pretrained_from='balanced.pth'):
    model = CRNN10(inputdim, outputdim)
    if pretrained_from:
        state = torch.load(pretrained_from,
                           map_location='cpu')
        model.load_state_dict(state, strict=False)
    return model


def mobilenetv2(inputdim=64, outputdim=527, pretrained_from='balanced.pth'):
    model = MobileNetV2_DM(inputdim, outputdim)
    if pretrained_from:
        state = torch.load(pretrained_from,
                           map_location='cpu')
        model.load_state_dict(state, strict=False)
    return model


def crnn(inputdim=64, outputdim=527, pretrained_from='balanced.pth'):
    model = CRNN(inputdim, outputdim)
    if pretrained_from:
        state = torch.load(pretrained_from,
                           map_location='cpu')
        model.load_state_dict(state, strict=False)
    return model


def cnn10(inputdim=64, outputdim=527, pretrained_from='balanced.pth'):
    model = CNN10(inputdim, outputdim)
    if pretrained_from:
        state = torch.load(pretrained_from,
                           map_location='cpu')
        model.load_state_dict(state, strict=False)
    return model


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""

    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(self.transform(logits))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),  # 归一化
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            # [batch, 1, time, dim] --> [batch, 32, time, dim]
            Block2D(1, 32),
            # [batch, 32, time, dim] --> [batch, 32, time / 2, dim / 4]
            nn.LPPool2d(4, (2, 4)),
            # [batch, 32, time / 2, dim / 4] --> [batch, 128, time / 2, dim / 4]
            Block2D(32, 128),
            # [batch, 128, time / 2, dim / 4] --> [batch, 128, time / 2, dim / 4]
            Block2D(128, 128),
            # [batch, 128, time / 2, dim / 4] --> [batch, 128, time / 4, dim / 16]
            nn.LPPool2d(4, (2, 4)),
            # [batch, 128, time / 4, dim / 16] --> [batch, 128, time / 4, dim / 16]
            Block2D(128, 128),
            # [batch, 128, time / 4, dim / 16] --> [batch, 128, time / 4, dim / 16]
            Block2D(128, 128),
            # [batch, 128, time / 4, dim / 64] --> [batch, 128, time / 4, dim / 64]
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        # [batch, 1, time, dim] --> [batch, 128, time/4, dim/64]
        with torch.no_grad():
            # [1, 1, 125, 1]
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
            inputdim=256,
            outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, _, time, dim = x.shape
        # 升维
        # x = x.unsqueeze(1)  # [batch, time, dim] --> [batch, 1, time, dim]
        # [batch, 1, time, dim] --> [batch, 128, time/4, 1]
        x = self.features(x)
        # print("self.features(x).shape : ", x.shape)
        # [batch, 128, time/4, dim/64] --> [batch, time/4, 128, dim/64] --> [batch, time/4, 2 * dim]
        x = x.transpose(1, 2).contiguous().flatten(-2)
        # print("self.features(x).shape 2 : ", x.shape)
        # [batch, time/4, 2 * dim] --> [batch,  time/4, 4 * dim] == [batch,  time/4, 256]
        x, _ = self.gru(x)
        # print("self.gru(x).shape : ", x.shape)
        # [batch,  time/4, 256] --> [batch, time/4, output_dim]
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        # decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        # decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.)

        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        # 上采样: [batch, time/4, output_dim] --> [batch, time, output_dim]

        # decision shape: [batch, output_dim]
        # decision_time shape: [batch, time, output_dim]
        return decision_time


class CNN10(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 64),  # [batch, 1, time, dim] --> [batch, 64, time, dim]
            # [batch, 64, time, dim] --> [batch, 64, time, dim]
            Block2D(64, 64),
            # [batch, 1, time, dim] --> [batch, 64, time / 2, dim / 4]
            nn.LPPool2d(4, (2, 4)),
            # [batch, 64, time / 2, dim / 4] --> [batch, 128, time / 2, dim / 4]
            Block2D(64, 128),
            # [batch, 128, time / 2, dim / 4] --> [batch, 128, time / 2, dim / 4]
            Block2D(128, 128),
            # [batch, 128, time / 2, dim / 4] --> [batch, 128, time / 4, dim / 8]
            nn.LPPool2d(4, (2, 2)),
            # [batch, 128, time / 4, dim / 8] --> [batch, 256, time/4, dim/8]
            Block2D(128, 256),
            # [batch, 256, time / 4, dim / 8] --> [batch, 256, time/4, dim/8]
            Block2D(256, 256),
            # [batch, 256, time / 4, dim / 8] --> [batch, 256, time/4, dim/16]
            nn.LPPool2d(4, (1, 2)),
            # [batch, 256, time/4, dim/16] --> [batch, 512, time/4, dim/16]
            Block2D(256, 512),
            # [batch, 512, time/4, dim/16] --> [batch, 512, time/4, dim/16]
            Block2D(512, 512),
            # [batch, 512, time/4, dim/16] --> [batch, 512, time/4, dim/64]
            nn.LPPool2d(4, (1, 2)),
            nn.Dropout(0.3),
            # [batch, 512, time/4, dim/64] --> [batch, 512, time/4, 1]
            nn.AdaptiveAvgPool2d((None, 1)),
        )

        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'attention'),
            inputdim=512,
            outputdim=outputdim)
        self.outputlayer = nn.Linear(512, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        # x: [batch, time, dim]
        batch, _, time, dim = x.shape
        # # x: [batch, 1, time, dim]
        # x = x.unsqueeze(1)
        # # [batch, 1, time, dim] --> [batch, 512, time/4, 1]
        # x = self.features(x)
        # # [batch, 512, time/4, 1] --> [batch, time/4, 512]
        # x = x.transpose(1, 2).contiguous().flatten(-2)
        # # [batch, time/4, 512] --> [batch, time/4, output_dim]
        # decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        # decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        # if upsample:
        #     decision_time = torch.nn.functional.interpolate(
        #         decision_time.transpose(1, 2),
        #         time,
        #         mode='linear',
        #         align_corners=False).transpose(1, 2)
        # # decision shape: [batch, output_dim]
        # # decision_time shape: [batch, time, output_dim]

        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision_time = torch.nn.functional.interpolate(
            decision_time.transpose(1, 2),
            time,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        return decision_time


class CRNN10(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self._hiddim = kwargs.get('hiddim', 256)
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 4)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (1, 2)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 2)),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(rnn_input_dim,
                          self._hiddim,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
            inputdim=self._hiddim*2,
            outputdim=outputdim)

        self.outputlayer = nn.Linear(self._hiddim*2, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        # batch, time, dim = x.shape
        # x = x.unsqueeze(1)
        # x = self.features(x)
        # x = x.transpose(1, 2).contiguous().flatten(-2)
        # decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        # decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        # if upsample:
        #     decision_time = torch.nn.functional.interpolate(
        #         decision_time.transpose(1, 2),
        #         time,
        #         mode='linear',
        #         align_corners=False).transpose(1, 2)

        batch, _, time, dim = x.shape
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        return decision_time


# def init_weights(m):
#     if isinstance(m, (nn.Conv2d, nn.Conv1d)):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)


class _ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(_ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), norm_layer(out_planes),
            nn.ReLU6(inplace=True))


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        layer: torch.nn.Sequential,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        layer = _ConvBNReLU if layer is None else layer

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                layer(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            layer(hidden_dim, hidden_dim, stride=stride,
                  groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_DM(nn.Module):

    def __init__(self,
                 outputdim=2,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 norm_layer=None,
                 **kwargs):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2_DM, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = kwargs.get('last_channel', 1280)

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [
            _ConvBNReLU(1, input_channel, stride=2, norm_layer=norm_layer)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    _InvertedResidual(input_channel,
                                      output_channel,
                                      stride,
                                      expand_ratio=t,
                                      layer=_ConvBNReLU,
                                      norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            _ConvBNReLU(input_channel,
                        self.last_channel,
                        kernel_size=1,
                        norm_layer=norm_layer))
        features.append(nn.AdaptiveAvgPool2d((1, None)))

        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.last_channel, outputdim),
        )

        # weight initialization
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        # x = B N
        # if self.training:
        #     x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        # x = self.front_end(x)
        # # B F T
        # if self.training:
        #     x = self.spectransforms(x)
        # x: [batch, time, dim] --> [batch, 1, dim, time]
        # x = rearrange(x, 'b t f -> b 1 f t')  # Add channel dim

        # x = rearrange(x, 'b t f -> b t f 1')
        # # [batch, 1, dim, time] -->
        # x = self.features(x)
        # x = rearrange(x, 'b c f t -> b (f t) c')
        # x = torch.sigmoid(self.classifier(x))

        # x: [batch, time, dim, 1]

        x = rearrange(x, 'b f t -> b 1 f t')  # Add channel dim
        x = self.features(x)
        print("shape: ", x.shape)
        x = rearrange(x, 'b c f t -> b (f t) c')
        print("shape: ", x.shape)
        x = torch.sigmoid(self.classifier(x))
        return x.mean(1), x


class DNN(nn.Module):
    def __init__(self, inputdim=64, outputdim=2, hidden_size=128, **kwargs) -> None:
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(inputdim, hidden_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc1_drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc2_drop = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc3_drop = nn.Dropout(p=0.2)

        self.last = nn.Linear(hidden_size, outputdim)

    def forward(self, x):
        # x : [batch, 1, time, dim]
        x = x.squeeze(1)
        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))

        out = self.last(out)

        return out


if __name__ == "__main__":
    inp = torch.randn(32, 1, 32, 64)
    model = DNN()
    output = model(inp)
    print(output.shape)
