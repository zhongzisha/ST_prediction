
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by the num_heads"

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, attention_mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attention_mask is not None:
            attention_scores = attention_mask.masked_fill(attention_mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_scores, V)
        return output

    def forward(self, Q, K, V, attention_mask=None):
        batch_size, seq_length, dim = Q.size()
        Q = self.W_q(Q)  # BLD
        K = self.W_k(K)
        V = self.W_v(V)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  # BHLd
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, attention_mask=attention_mask) # BHLd
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)

        output = self.W_o(attention_output)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)).float() * -(math.log(10000.0)/d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # 1 x L x D

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        attention_output = self.self_attention(x, x, x, attention_mask=attention_mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))
        cross_output = self.cross_attention(attention_output, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedding = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedding = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        enc_output = src_embedding
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        dec_output = tgt_embedding
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output



def batch_norm(x, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-5):

    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)

    running_mean = momentum * running_mean + (1-momentum) * batch_mean
    running_var = momentum * running_var + (1-momentum) * batch_var

    x_normalized = (x - batch_mean)  / torch.sqrt(batch_var + eps)

    output = weight * x_normalized + bias
    return output, running_mean, running_var

def layer_norm(x, epsilon=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - mean) / torch.sqrt(var+eps)
    return x_normalized

def cross_entropy_loss(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
    log_probs = torch.log(y_pred)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()
    loss = -torch.sum(y_true_one_hot*log_probs, dim=1)
    return loss.mean()



def bce_loss(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
    bce = -(y_true * torch.log(y_pred) + (1-y_true)*torch.log(1-y_pred))

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


