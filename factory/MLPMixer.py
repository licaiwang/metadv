import torch.nn as nn
from functools import partial


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, int(dim * expansion_factor)),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(int(dim * expansion_factor), dim),
        nn.Dropout(dropout),
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def MLPMixer(
    in_chans,
    out_chans,
    seq_len=176,
    depth=1,
    kernel_size=5,
    padding=2,
    expansion_factor=1,
    dropout=0.0,
):
    chan_first, chan_last = (
        partial(nn.Conv1d, kernel_size=1),
        partial(nn.Conv1d, kernel_size=kernel_size, padding=padding),
    )
    return nn.Sequential(
        nn.Linear(seq_len, seq_len),
        *[
            nn.Sequential(
                PreNormResidual(
                    seq_len,
                    FeedForward(in_chans, expansion_factor, dropout, chan_first),
                ),
                PreNormResidual(
                    seq_len, FeedForward(in_chans, expansion_factor, dropout, chan_last)
                ),
            )
            for _ in range(depth)
        ],
        nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
    )
