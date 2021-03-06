# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        # X: [batch_size, num_steps, embed_size]
        X = self.embedding(X)
        # GRU: 时间步在第一维度
        X = X.permute(1, 0, 2)
        # outputs:[num_steps, batch_size, num_hidden], state:[num_layers, batch_size, num_hiddens]
        outputs, state = self.rnn(X)
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 把Encoder的输出+Decoder的Embedding作为Decoder的输入
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, encoder_outputs):
        # encoder最后输出的state
        return encoder_outputs[1]

    def forward(self, X, state):
        # X:[num_steps, batch_size, embed_size]
        X = self.embedding(X).permute(1, 0, 2)
        # 将最后一层state作为context，采用广播机制，使其具有与X相同的num_steps;
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output:[batch_size, num_steps, vocab_size], state: [num_layers, batch_size, num_hiddens]
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    max_len = X.size(1)
    mask = torch.arange((max_len), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    # 取反
    X[~mask] = value
    return X


class MaskedSoftmaxCEloss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉上损失函数
    pred:[batch_size, num_steps, vocab_size]
    label:[batch_size, num_steps]
    valid_len: [batch_size]
    """
    def forward(self, pred, label, valid_len):
        """<pad>不计入损失"""
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweight_loss = super(MaskedSoftmaxCEloss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweight_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCEloss()
    net.train()

    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_vaild_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 起始标志
            bos = torch.tensor([tgt_vocab["<bos>"]]*Y.shape[0], device=device).reshape(-1, 1)
            # 训练过程采用teacher force
            dec_input = torch.cat((bos, Y[:, :-1]), dim=1)
            Y_hat, _ = net(X, dec_input)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            optimizer.step()




if __name__ == '__main__':
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(sequence_mask(X, torch.tensor([1, 2])))