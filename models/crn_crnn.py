import torch
from torch import nn
from crn import CausalConvBlock, CausalTransConvBlock
from crnn import Block2D, init_weights, parse_poolingfunction


class CRN_CRNN(nn.Module):
    def __init__(self, input_dim=161, output_dim=2, **kwargs) -> None:
        super(CRN_CRNN, self).__init__()

        # CRN for SE
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        self.lstm_layer = nn.LSTM(
            input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(
            32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

        # CRNN for VAD
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
        # with torch.no_grad():
        #     # [1, 1, 125, 1]
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,
        #                                               input_dim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(2048,
                          256,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
            inputdim=256,
            outputdim=output_dim)
        self.outputlayer = nn.Linear(512, output_dim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        self.lstm_layer.flatten_parameters()

        e_1 = self.conv_block_1(x)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels *
                              n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]

        # CRNN for VAD [2, T, 1204] -> [2, 1, T, 1204]
        crnn_x = lstm_out.unsqueeze(1)
        batch, _, time, dim = crnn_x.shape
        crnn_x = self.features(crnn_x)

        crnn_x = crnn_x.transpose(1, 2).contiguous().flatten(-2)
        crnn_x, _ = self.gru(crnn_x)
        decision_time = torch.sigmoid(self.outputlayer(crnn_x)).clamp(1e-7, 1.)

        decision_time = torch.nn.functional.interpolate(decision_time.transpose(1, 2),
                                                        time,
                                                        mode='linear',
                                                        align_corners=False).transpose(1, 2)

        lstm_out = lstm_out.permute(0, 2, 1).reshape(
            batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]

        # for SE
        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))

        return (decision_time, d_5)


if __name__ == "__main__":
    model = CRN_CRNN()
    inp = torch.randn((2, 1, 161, 200))
    out = model(inp)
    print(out[0].shape, out[1].shape)
