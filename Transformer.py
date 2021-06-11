# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import copy


# https://nlp.seas.harvard.edu/2018/04/03/attention.html


# 词向量embedding
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


# position encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 位置编码： max_len*d_model
        pe = torch.zeros(max_len, d_model)
        # 位置信息：max_len*1  [0,1,2...max_len]
        position = torch.arange(0, max_len).unsqueeze(1)
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        # 位置计算公式中除数：10000**（2i/d_model)
        # div_term: 1*(d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.)/d_model))

        # 偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch 维度  1*max_len*d_model
        pe = pe.unsqueeze(0)
        # pe作为模型参数保存，并不参与训练
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 按照输入文本最大长度进行位置编码截取
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    """用于克隆多个相同模块"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# scaled dot-product attention
def attention(query, key, val, mask=None, dropout=None):
    """
    实现 scaled dot-product attention
    ATT(Q, K, V) = softmax(Q*transpose(K) / sqrt(d_k)) * V

    :param query: 与Q矩阵相乘后的结果 size: (batch, h, L, d_model//h)
    :param key: 与K相乘结果, size同上
    :param val: 与V相乘结果， size同上
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask:
        # 掩码矩阵，编码器mask的size:(batch, 1, 1, src_L)
        # 解码器的size:(batch, 1, tag_L, tag_L)
        scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))
    # (batch, h, L, L)
    p_att = F.softmax(scores, dim=-1)
    if dropout:
        p_att = dropout(p_att)
    return torch.matmul(p_att, val), p_att


# 多头attention
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        """
        多头注意力机制
        :param h: 头数
        :param d_model: word embedding维度
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "头数必须被word embedding维度整除"
        self.d_k = d_model // h
        self.h = h
        # 四个线性变换，前三个为QKV变换矩阵，最后一个用于attention之后
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch, L, d_model)
        :param key: 同上
        :param value: 同上
        :return:
        """
        if mask:
            # 增加head维度，针对每个head都要mask
            mask = mask.unsqueeze(1)
        # batch大小
        nbatches = query.size(0)

        # 1. 利用全连接计算QKV矩阵，在进行维度变换[batch, L, d_model] -> [batch, h, L, d_k]
        # 这里的维度变换使转换后的矩阵变成多个头
        query, key, val = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        # 2. attention; x:[batch, h, L, d_model//h], self.attn: [batch, h, L, L]
        x, self.attn = attention(query, key, val, mask=mask, dropout=self.dropout)

        # 3. 拼接attention后的矩阵; [batch, h, L, d_model//h] -> [batch, L, h, d_model//h] -> [batch, L, d_model]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)

        # 4. 对拼接后的矩阵进行转换
        return self.linears[-1](x)


class Batch:
    def __init__(self, src, trg=None, pad=0):
        """
        生成mask
        :param src: 一个batch输入, [batch, src_L]
        :param trg: 一个batch输出, [batch, trg_L]
        :param pad: <pad>索引
        """
        self.src = src
        # src_mask [batch, 1, src_L]
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg:
            # 用于输入，不包含结尾<eos>
            self.trg = trg[:, :-1]
            # 用于输出计算损失，不包含起始<sos>
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # [batch, 1, tgt_L]
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt_mask.size(-1)).type_as(tgt_mask.data)
        )
        return tgt_mask

def subsequent_mask(size):
    """生成下三角矩阵，下三角为True"""
    att_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(att_shape), k=1).astype("uint8")
    return torch.from_numpy(subseq_mask) == 0



# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """层归一化"""
        super(LayerNorm, self).__init__()
        # 对归一化后的结果进行不同程度的偏移
        self.a_2 = nn.Parameter(torch.ones(features))
        # 偏置
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 防止方差为0
        self.eps = eps


    def forward(self, x):
        """x [batch, L, d_model]"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


# 残差链接
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# 前馈层, 全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        """FFN(x)=max(0,xW1+b1)W2+b2"""
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))



# 编码层
class EncodeLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncodeLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 两层残差，一层在self_att后， 一层在feed_forward后
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # multihead + resconnection => [batch, L, d_model]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # feedforward + resconnection => [batch, L, d_model]
        return self.sublayer[1](x, self.feed_forward)


# Encoder, 封装多个encoder layer
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class DecoderLayer(nn.Module):
    """self_attention + encoder-decoder-self_att + feed_forward"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)


    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x: target  [batch, tgt_L, d_model]
        :param memory: encoder输出  [batch, src_L, d_model]
        :param src_mask: [batch, 1, src_L]
        :param tgt_mask: [batch, tgt_L, tgt_L]
        :return:
        """
        m = memory
        # self_att, add&norm [batch, tgt_L, d_model]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # encoder-decoder-att, add&norm, Q来自target， KV来自encoder output; [batch, tgt_L, d_model]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # feed_forward, add&norm  [batch, tgt_L, d_model]
        return self.sublayer[2](x, self.feed_forward)



class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # [batch, tgt_L, d_model]
        return self.norm(x)



class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        # 编码器
        self.encoder = encoder
        # 解码器
        self.decoder = decoder
        # 源 embedding
        self.src_embed = src_embed
        # target embedding
        self.tgt_embed = tgt_embed
        # 解码器后的线性变换和softmax
        self.generator = generator


    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)




def make_model(src_vocab, tgt_vocab, N=6, d_model=512,
               d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncodeLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model






























