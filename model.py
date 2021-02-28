import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        sequence_len,
        d_model,
        dim_feedforward,
        nhead,
        num_layers,
        dropout
    ):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.sequence_len = sequence_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            custom_decoder=DummyDecoder()
        )
        self.decoder = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def gen_mask(self, size):
        return self.transformer.generate_square_subsequent_mask(size)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
        if mask is None:
            device = next(self.parameters()).device
            mask = self.gen_mask(src.shape[1]).to(device)
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # (max_seq, batch_size, d_model)
        src = self.pos_encoder(src)
        out = self.transformer(src=src, tgt=src, src_mask=mask)
        out = out.permute(1, 0, 2)  # (batch_size, max_seq, d_model)
        out = self.decoder(out)
        return out


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, *args, **kwargs):
        return memory
