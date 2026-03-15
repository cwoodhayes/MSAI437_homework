import argparse

# import os
# import sys
# import shutil
import random
import numpy as np
from collections import deque

# import time
import copy
import math
# import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sns


# def OutText(text,opt,screen=True):
#    if screen:
#        print(text)
#    if opt.log_file:
#        outFile = open(opt.log_file,"a+")
#        outFile.write(text+"\n")

last_scores = None
last_output = None
last_10_scores = deque(maxlen=10)


def read_encode(file_name, vocab, words, corpus, threshold):
    wID = len(vocab)

    if threshold > -1:
        with open(file_name, 'rt') as f:
            for line in f:
                line = line.replace('\n', '')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:  # noqa: E722
                        elem = [wID, 0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID, 100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID, temp[t][1]]

    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:  # noqa: E722
                    wID = words['<unk>'][0]
                corpus.append(wID)

    return [vocab, words, corpus]


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / d_model))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(
            self.pe[:, :seq_len],  # type: ignore
            requires_grad=False,
        )
        pe = pe.to(x.device)
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_k: int,
    p: float,
    mask: torch.Tensor | None,
    dropout=None,
):
    """
    Attention mechanism used by MultiHeadAttention.

    Args:
        q: query vector. shape = (batch_len=1, n_heads=1, seq_len=20, d_k)
        k: key vector. shape = (batch_len=1, n_heads=1, seq_len=20, d_k)
        v: value vector. shape = (batch_len=1, n_heads=1, seq_len=20, d_k)
        d_k: dimension of each head
        p: ignored.
        mask: attention mask. shape =
            (batch_len=1, n_heads=1, seq_len=20, seq_len=20)
        dropout: ignored.

    Returns:
        torch.Tensor of shape (batch_len=1, n_heads=1, seq_len=20, d_k).
        this is "sum" in slides.

    """
    global last_output
    global last_scores

    seq_len = q.shape[2]
    scores = torch.zeros((1, 1, seq_len, seq_len), device=q.device)
    output = torch.zeros((1, 1, seq_len, d_k), device=q.device)
    if mask is None:
        # if no mask, everything passed through
        mask = torch.ones_like(scores)
    else:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=q.device, dtype=torch.float32)

    # z = softmax(KQ^T / sqrt(d_k)) @ V
    # doing this in for loops per assignment
    # first let's get input to softmax (k dot q / sqrt)
    sm_input = torch.zeros((1, 1, seq_len, seq_len), device=q.device)
    for token_idx in range(q.shape[2]):
        for other_tok_idx in range(q.shape[2]):
            dot = 0
            for val_idx in range(q.shape[3]):
                dot += (
                    q[0][0][token_idx][val_idx]
                    * k[0][0][other_tok_idx][val_idx]
                )
            scaled = dot / math.sqrt(d_k)
            if mask[0][0][token_idx][other_tok_idx] == 0:
                # -inf
                scaled = -1e9
            sm_input[0][0][token_idx][other_tok_idx] = scaled

    # now take softmax. each row is probability weighting for a token,
    # across all tokens.
    for token_idx in range(sm_input.shape[2]):
        denom = 0
        for other_tok_idx in range(sm_input.shape[2]):
            denom += torch.exp(sm_input[0][0][token_idx][other_tok_idx])
        for other_tok_idx in range(sm_input.shape[2]):
            scores[0][0][token_idx][other_tok_idx] = (
                torch.exp(sm_input[0][0][token_idx][other_tok_idx]) / denom
            )

    # multiply softmax by V matrix.
    # this weights the values by the sm outputs
    # and then sum all the values for each token, elementwise
    for token_idx in range(output.shape[2]):
        for other_tok_idx in range(scores.shape[3]):
            sm_weight = scores[0][0][token_idx][other_tok_idx]
            for val_idx in range(d_k):
                output[0][0][token_idx][val_idx] += (
                    sm_weight * v[0][0][other_tok_idx][val_idx]
                )

    last_scores = scores
    last_output = output
    last_10_scores.append(scores)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, seqlen, norm, opt, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.sigma = torch.ones([seqlen, seqlen], dtype=torch.float32)
        self.sigma = self.sigma.cuda()
        self.norm = norm
        self.opt = opt

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, self.norm, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        factor: float = 1.0,
    ) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.0
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called,
        # since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min
                + ((lr - self.eta_min) / 2)
                * (
                    np.cos(
                        np.pi
                        * ((self._cycle_counter) % self._updated_cycle_len)
                        / self._updated_cycle_len
                    )
                    + 1
                )
            )
            for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


class DecoderLayerGPT(nn.Module):
    def __init__(self, d_model, heads, seqlen, norm, opt, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(
            heads, d_model, seqlen, norm, opt, dropout=dropout
        )
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderGPT(nn.Module):
    def __init__(
        self, vocab_size, d_model, N, heads, seqlen, norm, opt, dropout
    ):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(
            DecoderLayerGPT(d_model, heads, seqlen, norm, opt, dropout), N
        )
        self.norm = Norm(d_model)

    def forward(self, trg, mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerGPT(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, opt):
        super().__init__()
        self.decoder = DecoderGPT(
            vocab_size, d_model, N, heads, opt.seqlen, opt.norm, opt, dropout
        )
        self.decoder = self.decoder.cuda()
        self.out = nn.Linear(d_model, vocab_size)
        self.out = self.out.cuda()
        self.opt = opt

    def forward(self, trg, trg_mask):
        d_output = self.decoder(trg, trg_mask)
        if self.opt.tied == 0:
            output = self.out(d_output)
        else:
            output = torch.matmul(
                d_output, self.decoder.embed(self.opt.indices).transpose(0, 1)
            )

        return [d_output, output]


class MyHead(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.head = nn.Linear(dims, 1)
        self.head = self.head.cuda()

    def forward(self, cls):
        result = self.head(cls)
        return result


def get_modelGPT(opt, vocab_size):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = TransformerGPT(
        vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt
    )

    if opt.loadname is not None:
        print('loading pretrained weights...')
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if opt.device == 0:
        model = model.cuda()

    return model


def plot_attention(
    scores: np.ndarray,  # 20x20 array
    token_labels: list,
    title='Attention Heatmap',
    subtitle: str | None = None,
):
    """Plot a heatmap of attention scores."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        scores,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap='Blues',
        ax=ax,
    )
    ax.set_xlabel('Key (attending to)')
    ax.set_ylabel('Query (attending from)')
    actual_title = title
    if subtitle is not None:
        actual_title += '\n"' + subtitle + '"'
    ax.set_title(actual_title)
    plt.tight_layout()
    plt.savefig(f'{title}.png')


def plot_attention_multiple(
    scores_list: list[np.ndarray],  # list of 20x20 arrays
    token_labels: list,
    title='Attention Heatmap',
    subtitle: str | None = None,
):
    """
    Plot an average heatmap of multiple examples that share a common structure.

    This is for part 2. Basically any token that's the same across all examples
    will be labeled with that token label, and anything that's not
    identical across all with be shown as [X].

    Useful for showing model attention over many similar examples,
    eg all multiplications b * b
    """
    if len(scores_list) == 0:
        raise ValueError('scores_list must contain at least one score matrix')

    normalized_scores = []
    for scores in scores_list:
        scores_np = np.asarray(scores)
        if scores_np.ndim == 4:
            scores_np = scores_np[0, 0]
        elif scores_np.ndim == 3:
            scores_np = scores_np[0]
        normalized_scores.append(scores_np)

    avg_scores = np.mean(np.stack(normalized_scores, axis=0), axis=0)

    if len(token_labels) > 0 and isinstance(token_labels[0], list):
        labels_by_position = []
        for labels_at_pos in zip(*token_labels):
            first = labels_at_pos[0]
            if all(label == first for label in labels_at_pos):
                labels_by_position.append(first)
            else:
                labels_by_position.append('[X]')
    else:
        labels_by_position = token_labels

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        avg_scores,
        xticklabels=labels_by_position,
        yticklabels=labels_by_position,
        cmap='Blues',
        ax=ax,
    )
    ax.set_xlabel('Key (attending to)')
    ax.set_ylabel('Query (attending from)')
    ax.set_title(title, pad=16)
    if subtitle is not None:
        fig.suptitle(subtitle, y=0.98, fontsize=10)
        plt.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        plt.tight_layout()
    plt.savefig(f'{title}.png')


def select_good_and_bad_examples(opt) -> tuple[list[int], list[int]]:
    """
    Return first 10 good/bad b*b example indices from results.txt.

    good examples have correct-answer probability > 50%
    bad examples < 5%.
    """
    good = []
    bad = []
    high_thresh = 80.0
    low_thresh = 1.0

    with open('results.txt', 'rt') as f:
        for obs_idx, line in enumerate(f):
            if '[EQ] ans = b * b' not in line:
                continue

            try:
                prob = float(line.split()[-1].rstrip('%'))
            except (IndexError, ValueError):
                continue

            if prob > high_thresh and len(good) < 10:
                good.append(obs_idx)
            if prob < low_thresh and len(bad) < 10:
                bad.append(obs_idx)

            if len(good) == 10 and len(bad) == 10:
                break

    if len(good) < 10 or len(bad) < 10:
        raise RuntimeError(
            'Could not find enough examples in results.txt '
            f'(good={len(good)}, bad={len(bad)})'
        )

    return good, bad


def example(model, opt):
    model.eval()

    # good = [1067, 701, 1979, 1005, 2041, 658, 1740, 606, 1707, 42]
    # bad = [946, 1322, 2487, 314, 1445, 127, 1959,
    # 2344, 1947, 2105, 1441, 885]
    good, bad = select_good_and_bad_examples(opt)

    aa = opt.seqlen
    bb = 1
    opt.bb = bb
    offsets = []
    stride = int(len(opt.test) / bb)
    while (stride % 20) > 0:
        stride = stride - 1
    for i in range(0, len(opt.test), stride):
        offsets.append(i)

    nopeak_mask = np.triu(np.ones((bb, aa, aa), dtype=np.int32), k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()

    print('GOOD Examples:')
    avg = 0.0
    good_token_labels = []
    good_example_text = []
    for i in good:
        trg = torch.zeros((bb, aa), dtype=torch.long)
        ans = torch.zeros((bb, opt.vocab_size), dtype=torch.float)
        text = ''
        for j in range(aa):
            trg[0, j] = opt.test[i * aa + j]
            if j == 19:
                ans[0, trg[0, j]] = 1.0
            text = decode_formula(opt, trg)
        good_token_labels.append(
            [opt.vocab[int(token_id)] for token_id in trg[0]]
        )
        good_example_text.append(text)
        trg = trg.cuda()
        ans = ans.cuda()

        [d_output, preds] = model(trg, mask)
        logits = torch.exp(preds[:, 18, :])
        numer = logits * ans
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(logits, dim=1)
        probs = numer / denom
        print('%s %7.3f%%' % (text, 100.0 * probs[0].item()))
        avg = avg + probs[0].item()
    print(
        '                                                                       Average: %7.3f%%'  # noqa: E501
        % (100.0 * avg / float(len(good)))
    )
    print(' ')

    if (
        last_scores is not None
        and last_output is not None
        and good_token_labels
        and good_example_text
    ):
        plot_attention(
            last_scores[0, 0].detach().cpu().numpy(),
            token_labels=good_token_labels[-1],
            title='Attention Heatmap for GOOD Example',
            subtitle=good_example_text[-1],
        )
        scores = [sc.detach().cpu().numpy() for sc in last_10_scores]
        avg_prob = 100.0 * avg / float(len(good))
        plot_attention_multiple(
            scores_list=scores,
            token_labels=good_token_labels,
            title='Average Attention Heatmap for GOOD Examples',
            subtitle=f'Average of 10 examples (avg prob. of correct ans = {avg_prob:7.3f}%)',  # noqa: E501
        )
    else:
        raise RuntimeError('should be unreachable')

    print('BAD Examples:')
    avg = 0.0
    bad_token_labels = []
    bad_example_text = []
    for i in bad:
        trg = torch.zeros((bb, aa), dtype=torch.long)
        ans = torch.zeros((bb, opt.vocab_size), dtype=torch.float)
        text = ''
        for j in range(aa):
            trg[0, j] = opt.test[i * aa + j]
            if j == 19:
                ans[0, trg[0, j]] = 1.0
            text = decode_formula(opt, trg)
        bad_token_labels.append(
            [opt.vocab[int(token_id)] for token_id in trg[0]]
        )
        bad_example_text.append(text)
        trg = trg.cuda()
        ans = ans.cuda()

        [d_output, preds] = model(trg, mask)
        logits = torch.exp(preds[:, 18, :])
        numer = logits * ans
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(logits, dim=1)
        probs = numer / denom
        top3_values, top3 = torch.topk(logits, k=3)
        print(
            '%s %7.3f%% %5s %5s %5s'
            % (
                text,
                100.0 * probs[0].item(),
                opt.vocab[top3[0, 0]],
                opt.vocab[top3[0, 1]],
                opt.vocab[top3[0, 2]],
            )
        )
        avg = avg + probs[0].item()
    print(
        '                                                                       Average: %7.3f%%'  # noqa: E501
        % (100.0 * avg / float(len(bad)))
    )
    print(' ')

    if (
        last_scores is not None
        and last_output is not None
        and bad_token_labels
        and bad_example_text
    ):
        plot_attention(
            last_scores[0, 0].detach().cpu().numpy(),
            token_labels=bad_token_labels[-1],
            title='Attention Heatmap for BAD Example',
            subtitle=bad_example_text[-1],
        )
        scores = [sc.detach().cpu().numpy() for sc in last_10_scores]
        avg_prob = 100.0 * avg / float(len(bad))
        plot_attention_multiple(
            scores_list=scores,
            token_labels=bad_token_labels,
            title='Average Attention Heatmap for BAD Examples',
            subtitle=f'Average of 10 examples (avg prob. of correct ans = {avg_prob:7.3f}%)',  # noqa: E501
        )
    else:
        raise RuntimeError('should be unreachable')

    plt.show()


def decode_formula(opt, trg):
    vocab = opt.vocab
    text = '[START] a %5s b %5s c %5s d %5s ' % (
        vocab[trg[0, 2]],
        vocab[trg[0, 4]],
        vocab[trg[0, 6]],
        vocab[trg[0, 8]],
    )
    text = text + '[VARS] %s %s [EQ] ans = %s %s %s [ANS] %5s' % (
        vocab[trg[0, 10]],
        vocab[trg[0, 11]],
        vocab[trg[0, 15]],
        vocab[trg[0, 16]],
        vocab[trg[0, 17]],
        vocab[trg[0, 19]],
    )
    return text


def run_single(model, opt, obs_idx):
    """Run on just one observation. Used for part 1."""

    def write_matrix(file_name, matrix):
        with open(file_name, 'w') as f:
            for row in matrix:
                values = ','.join(f'{value:.5f}' for value in row)
                f.write(values + ',\n')

    model.eval()
    aa = opt.seqlen
    bb = 1

    nopeak_mask = np.triu(np.ones((bb, aa, aa), dtype=np.int32), k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0).cuda()

    trg = torch.zeros((bb, aa), dtype=torch.long)
    for j in range(aa):
        trg[0, j] = opt.test[obs_idx * aa + j]
    text = decode_formula(opt, trg)
    trg = trg.cuda()

    with torch.no_grad():
        [d_output, preds] = model(trg, mask)

    print(text)

    if last_scores is not None and last_output is not None:
        scores_np = last_scores[0, 0].detach().cpu().numpy().round(5)
        output_np = last_output[0, 0].detach().cpu().numpy().round(5)
        write_matrix(
            f'cwh_{obs_idx}_scores_{opt.d_model}.csv',
            scores_np,
        )
        write_matrix(
            f'cwh_{obs_idx}_output_{opt.d_model}.csv',
            output_np,
        )
    else:
        raise RuntimeError('unreachable')


def main():
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=0)
    parser.add_argument('-savename', type=str)
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str, default='model')
    parser.add_argument('-norm', type=float, default=0.0)
    parser.add_argument('-run_single', action='store_true')
    parser.add_argument(
        '-idx',
        type=int,
        default=0,
        help='index of single observation to run on if -run_single is set',
    )

    opt = parser.parse_args()
    opt.verbose = False

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device('cuda:0')

    [opt.vocab, opt.words, opt.train] = read_encode('train.txt', [], {}, [], 0)
    print('vocab: %d train: %d' % (len(opt.vocab), len(opt.train)))
    [opt.vocab, opt.words, opt.test] = read_encode(
        'test.txt', opt.vocab, opt.words, [], -1
    )
    print('vocab: %d test: %d' % (len(opt.vocab), len(opt.test)))
    [opt.vocab, opt.words, opt.valid] = read_encode(
        'valid.txt', opt.vocab, opt.words, [], -1
    )
    print('vocab: %d test: %d' % (len(opt.vocab), len(opt.test)))

    opt.vocab_size = len(opt.vocab)
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()

    model = get_modelGPT(opt, opt.vocab_size)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    text = 'total params: %d' % (params)
    print(text)

    if opt.run_single:
        print(f'Running on OBSERVATION {opt.idx}...')
        run_single(model, opt, obs_idx=opt.idx)
        return
    else:
        example(model, opt)


if __name__ == '__main__':
    main()
