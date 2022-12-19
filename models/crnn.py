import torch
from torch import nn


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
    def __init__(self, inputdim=64, outputdim=2, **kwargs):
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


if __name__ == "__main__":
    model = CRNN()
    inp = torch.randn(1, 1, 10, 64)
    out = model(inp)
    print(out.shape)
