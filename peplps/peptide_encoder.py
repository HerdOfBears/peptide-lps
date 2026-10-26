"""Standalone peptide encoders: sequence -> latent vector.

Frozen encoders from a study of hierarchical autoencoders for antimicrobial peptide
(AMP) latent spaces, trained on 243,195 peptides. Two architectures are shipped, at
seven latent widths each:

    hvae_transvae_dNNN_encoder.pt   hierarchical Transformer VAE, returns the
                                    posterior mean mu
    hvae_wae_dNNN_encoder.pt        hierarchical Wasserstein autoencoder, returns
                                    its deterministic code

Each file holds one encoder and its 2-unit AMP property head -- no decoders, no
optimizer state, and no part of the original ``transvae`` package. Depends on
``torch`` and ``numpy`` only.

Usage as a library::

    from peptide_encoder import PeptideEncoder

    enc = PeptideEncoder.load("encoders/hvae_transvae_d128_encoder.pt")
    Z, amp = enc.encode(["GIGKFLHSAKKFGKAFVGEIMNS", "ACDEFGHIKL"])

Usage from the shell::

    python peptide_encoder.py --weights encoders/hvae_wae_d064_encoder.pt \
        --in peptides.fasta --out embeddings.npz

See README.md for the input contract, and for how to use the output in a Bayesian
optimization loop -- the latent coordinates need standardizing first, and the
search box must come from encoded data rather than from the prior.

GENERATED FILE -- do not edit. It is emitted by ``handoff/build_encoder_package.py``
in the source repository; edit that and rebuild. The model classes below are copied
verbatim from the training library, so this file reproduces the published embeddings
bit for bit; ``verify.py`` checks that on your machine.
"""

import argparse
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


# =========================================================================
# VERBATIM BLOCK -- copied byte for byte from the training library by
# handoff/build_encoder_package.py. Do not edit: these definitions are
# what make this file reproduce the published embeddings exactly.
#
#   transvae/trans_models.py
#     sha256 f9aebf28c4866f5f20dacf135060a9a35a0153838561c89357f9ad1099130c6e
#   transvae/transformer_models.py
#     sha256 1699964582dc1cf0710a1b417a1d5965f75c6e75bde854682e85927ab551422a
#   transvae/tvae_util.py
#     sha256 ea65f0a8baeccbb1a5e7c4381c2aa0d78f3488c1d33734b4e15b9e428c95d3d6
#   transvae/wae_models.py
#     sha256 f37ed028d2b5c096ae00922de305da1d30b984214c9ce6f2cf1724afb7f3362a
# =========================================================================

# ----- ListModule, verbatim from transvae/tvae_util.py -----
class ListModule(nn.Module):
    """Wrap a sequence of ``nn.Module`` instances so that they are registered
    as sub-modules (and thus visible to ``.parameters()``), while supporting
    integer indexing and iteration.

    Parameters
    ----------
    *args : nn.Module
        Modules to register.
    """
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

# ----- ConvBottleneck, verbatim from transvae/trans_models.py -----
class ConvBottleneck(nn.Module):
    """Three-stage 1-D convolutional bottleneck (encoder side).

    Each stage consists of ``Conv1d`` → ``MaxPool1d`` → ``ReLU``.  Channel counts
    taper from ``d_model`` down to 64 over the three stages.  The post-pool spatial
    lengths are recorded in ``self.conv_list`` (and the per-stage pool strides in
    ``self.strides``) so the matching :class:`DeconvBottleneck` can rebuild the
    exact inverse geometry.

    There are two layouts, selected by ``target_len``:

    ``target_len is None`` (legacy)
        The original schedule: each stage runs a no-pad ``Conv1d`` (preferred
        kernels ``[9, 8, 8]``, clamped to ``max(1, L - 1)`` and forced odd) followed
        by a stride-2 ``MaxPool``.  For ``src_len = 52`` this collapses
        ``52 → 22 → 8 → 1``, leaving a single spatial cell (``1 × 64 = 64`` features).
        This is kept for backward compatibility so checkpoints trained before the
        widening (which do not store ``TARGET_CONV_LEN``) still load unchanged.

    ``target_len = T`` (widened, the fix for short 52-token inputs)
        Land at *exactly* ``T`` spatial cells (``T × 64`` features) so the latent is
        not starved at ``d_latent`` 64/128 and positional ("where") information is
        preserved.  The 3-stage / 64-out-channel contract is unchanged:

        * Stages 0-1: a length-preserving ``'same'``-padded conv (odd kernel,
          ``padding = (k - 1) // 2``, stride 1) + stride-2 ``MaxPool`` → halve.
          E.g. ``52 → 26 → 13``.
        * Stage 2: a no-pad conv whose kernel is sized to reach exactly ``T``
          (``kernel = L2 - T + 1``), stride 1, with no pooling (recorded stride 1).
          E.g. ``13 → 8``.

        For ``src_len = 52`` the supported widths are ``T ∈ {1, 2, 4, 6, 8}`` →
        ``{64, 128, 256, 384, 512}`` features.  ``T`` must satisfy
        ``1 ≤ T ≤ src_len // 4`` (two halvings precede the final conv).

    Parameters
    ----------
    d_model : int
        Input channel count (encoder hidden size).
    src_len : int
        Input spatial length (source sequence length).
    target_len : int or None, optional
        If ``None`` (default) use the legacy schedule.  Otherwise build the widened
        schedule that lands at exactly ``target_len`` spatial cells.
    """
    def __init__(self, d_model, src_len, target_len=None):
        super().__init__()
        conv_layers = []
        self.conv_list = []
        self.strides = []
        self.kernels = []
        self.out_channels = 64

        in_c = d_model
        L = int(src_len)

        if target_len is None:
            # ---- legacy schedule (unchanged): no-pad conv + stride-2 pool x3 ----
            prefer_k = [9, 8, 8]   # same preference you used elsewhere

            for i in range(3):
                out_c = int((in_c - 64) // 2 + 64)
                if i == 2:
                    out_c = self.out_channels
                k_pref = prefer_k[i]
                k = min(k_pref, max(1, L - 1))
                if k > 1 and (k % 2 == 0):
                    k -= 1
                conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=1, padding=0)
                Lc = L - k + 1
                s = 2 if Lc >= 2 else 1
                pool = nn.MaxPool1d(kernel_size=s, stride=s, padding=0)
                Lp = Lc // s

                self.kernels.append(k)
                self.strides.append(s)
                self.conv_list.append(Lp)
                conv_layers.append(nn.Sequential(conv, pool))

                in_c = out_c
                L = Lp
        else:
            # ---- widened schedule: land at exactly target_len cells ----
            T = int(target_len)
            max_T = L // 4   # two stride-2 halvings happen before the final conv
            if T < 1 or T > max_T:
                raise ValueError(
                    f"target_len={T} is out of range for src_len={src_len}: "
                    f"need 1 <= target_len <= {max_T} (src_len // 4)."
                )

            prefer_k = [9, 7]   # odd kernels for the two 'same'-padded stages
            for i in range(2):
                out_c = int((in_c - 64) // 2 + 64)
                k = min(prefer_k[i], max(1, L - 1))
                if k % 2 == 0:
                    k -= 1                       # keep odd so 'same' padding is exact
                pad = (k - 1) // 2               # length-preserving ('same') conv
                conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=1, padding=pad)
                pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                Lp = L // 2                      # conv preserves length; pool halves

                self.kernels.append(k)
                self.strides.append(2)
                self.conv_list.append(Lp)
                conv_layers.append(nn.Sequential(conv, pool))

                in_c = out_c
                L = Lp

            # Final stage: a no-pad conv reaching exactly T, no pooling.
            k = L - T + 1                        # 13 - 8 + 1 = 6 for the default
            conv = nn.Conv1d(in_c, self.out_channels, kernel_size=k, stride=1, padding=0)
            pool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)   # no-op (stride 1)
            self.kernels.append(k)
            self.strides.append(1)
            self.conv_list.append(T)
            conv_layers.append(nn.Sequential(conv, pool))

        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for block in self.conv_layers:
            x = F.relu(block(x))
        return x

# ----- LayerNorm, verbatim from transvae/trans_models.py -----
class LayerNorm(nn.Module):
    """Manual layer normalisation.

    .. math::

        \\text{LayerNorm}(x) = a \\cdot \\frac{x - \\mu}{\\sigma + \\varepsilon} + b

    where ``a`` and ``b`` are learnable affine parameters.

    Parameters
    ----------
    features : int
        Last dimension of the input tensor.
    eps : float
        Small constant for numerical stability.

    References
    ----------
    Ba et al., "Layer Normalization", arXiv 2016.
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# ----- clones, verbatim from transvae/tvae_util.py -----
def clones(module, N):
    """Produce N identical (deep-copied) layers.

    Parameters
    ----------
    module : nn.Module
        Prototype layer to replicate.
    N : int
        Number of copies.

    Returns
    -------
    nn.ModuleList
        List of ``N`` independent copies of *module*.

    Notes
    -----
    Adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# ----- VAEEncoder, verbatim from transvae/transformer_models.py -----
class VAEEncoder(nn.Module):
    """Transformer encoder with convolutional bottleneck to latent space.

    Maps ``(B, L, d_model)`` attention outputs through a :class:`ConvBottleneck`
    to a flat vector, then projects to ``mu`` and ``logvar`` of shape
    ``(B, d_latent)`` for reparameterisation.

    Also contains a two-layer MLP that predicts the source mask length from
    ``mu``, enabling mask reconstruction during inference.

    Parameters
    ----------
    layer : EncoderLayer
        Prototype self-attention + FFN layer (will be cloned ``N`` times).
    N : int
        Number of encoder layers.
    d_latent : int
        Latent dimensionality.
    bypass_bottleneck : bool
        If ``True``, skip the stochastic bottleneck (plain autoencoder).
    eps_scale : float
        Scale factor for the reparameterisation noise.
    """
    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale, target_conv_len=None):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size, layer.src_len, target_conv_len)
        self.flat_conv_out = self.conv_bottleneck.conv_list[-1] * self.conv_bottleneck.out_channels
        self.z_means, self.z_var = nn.Linear(self.flat_conv_out, d_latent), nn.Linear(self.flat_conv_out, d_latent)
        self.norm = LayerNorm(layer.size)
        # Added to predict over source positions (including <start>?)
        self.max_src_len   = layer.src_len
        self.predict_len1 = nn.Linear(d_latent, d_latent*2)
        #self.predict_len2 = nn.Linear(d_latent*2, layer.size)
        self.predict_len2 = nn.Linear(d_latent*2, self.max_src_len)
        self.d_latent = d_latent
        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def predict_mask_length(self, mem):
        "Predicts mask length from latent memory so mask can be re-created during inference"
        pred_len = self.predict_len1(mem)
        pred_len = self.predict_len2(pred_len)
        pred_len = F.softmax(pred_len, dim=-1)
        pred_len = torch.topk(pred_len, 1)[1]
        return pred_len

    def reparameterize(self, mu, logvar, eps_scale=1):
        """Reparameterisation trick: ``z = mu + eps * sigma``.

        Allows gradients to flow through the stochastic sampling step by
        expressing the sample as a deterministic function of ``mu``,
        ``logvar``, and an auxiliary noise variable ``eps ~ N(0, I)``.

        References
        ----------
        Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std

    def forward(self, x, mask):
        ### Attention and feedforward layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        ### Batch normalization
        mem = self.norm(x)

        ### Zero out PAD rows before conv bottleneck.
        # Attention masking prevents valid tokens from attending to PAD tokens,
        # but PAD rows still accumulate non-zero values through residual
        # connections and feed-forward layers. Zero them so they don't
        # contaminate the latent representation via convolution.
        if mask is not None:
            if mask.dim() == 3:
                key_valid = mask.squeeze(1).bool()       # (B, T)
            elif mask.dim() == 2:
                key_valid = mask.bool()
            else:
                key_valid = None
            if key_valid is not None:
                mem = mem * key_valid.unsqueeze(-1).to(mem.dtype)

        ### Convolutional Bottleneck
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len

    def forward_w_attn(self, x, mask):
        "Forward pass that saves attention weights"
        attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mask, return_attn=True)
            attn_wts.append(wts.detach().cpu())
        mem = self.norm(x)

        # Zero out PAD rows before conv bottleneck (same as forward())
        if mask is not None:
            if mask.dim() == 3:
                key_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                key_valid = mask.bool()
            else:
                key_valid = None
            if key_valid is not None:
                mem = mem * key_valid.unsqueeze(-1).to(mem.dtype)

        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len, attn_wts

# ----- SublayerConnection, verbatim from transvae/trans_models.py -----
class SublayerConnection(nn.Module):
    """Pre-norm residual connection: ``x + Dropout(sublayer(LayerNorm(x)))``.

    This is the **pre-norm** variant where normalisation is applied *before*
    the sublayer (attention or FFN), not after.  This differs from the
    original Transformer paper's post-norm ordering but is commonly used
    for training stability.

    Parameters
    ----------
    size : int
        Model dimensionality.
    dropout : float
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with matching size."""
        return x + self.dropout(sublayer(self.norm(x)))

# ----- EncoderLayer, verbatim from transvae/transformer_models.py -----
class EncoderLayer(nn.Module):
    """Single Transformer encoder layer: self-attention + feed-forward, each
    wrapped in a pre-norm residual connection (:class:`SublayerConnection`).

    Parameters
    ----------
    size : int
        Model dimensionality (``d_model``).
    src_len : int
        Source sequence length (stored for downstream bottleneck sizing).
    self_attn : MultiHeadedAttention
    feed_forward : PositionwiseFeedForward
    dropout : float
    """
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

# ----- attention, verbatim from transvae/tvae_util.py -----
def attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention.

    .. math::

        \\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V

    Parameters
    ----------
    query : torch.Tensor
        Shape ``(B, h, T_q, d_k)`` or ``(B, T_q, d_k)``.
    key : torch.Tensor
        Shape ``(B, h, T_k, d_k)`` or ``(B, T_k, d_k)``.
    value : torch.Tensor
        Shape ``(B, h, T_k, d_v)`` or ``(B, T_k, d_v)``.
    mask : torch.Tensor, optional
        Broadcastable boolean mask; positions with ``0`` / ``False`` are filled
        with ``-1e9`` before softmax.
    dropout : nn.Dropout, optional
        Dropout layer applied to attention weights.

    Returns
    -------
    output : torch.Tensor
        Weighted values, same leading dims as *query* with last dim ``d_v``.
    p_attn : torch.Tensor
        Attention weight matrix (after softmax, before value multiplication).

    References
    ----------
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Eq. 1.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# ----- MultiHeadedAttention, verbatim from transvae/transformer_models.py -----
class MultiHeadedAttention(nn.Module):
    """Multi-head attention mechanism.

    Linearly projects queries, keys, and values into ``h`` parallel heads of
    dimensionality ``d_k = d_model / h``, applies scaled dot-product attention
    to each head independently, concatenates the results, and applies a final
    linear projection.

    Parameters
    ----------
    h : int
        Number of attention heads.
    d_model : int
        Total model dimensionality (must be divisible by ``h``).
    dropout : float
        Dropout rate applied to attention weights.

    References
    ----------
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Section 3.2.2.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)

# ----- PositionwiseFeedForward, verbatim from transvae/transformer_models.py -----
class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network.

    .. math::

        \\text{FFN}(x) = W_2 \\cdot \\text{ReLU}(W_1 x + b_1) + b_2

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    d_ff : int
        Hidden layer dimensionality.
    dropout : float
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ----- PositionalEncoding, verbatim from transvae/trans_models.py -----
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    For position *pos* and dimension *i*:

    .. math::

        PE_{(pos, 2i)}   &= \\sin\\bigl(pos / 10000^{2i/d_{\\text{model}}}\\bigr) \\\\
        PE_{(pos, 2i+1)} &= \\cos\\bigl(pos / 10000^{2i/d_{\\text{model}}}\\bigr)

    Parameters
    ----------
    d_model : int
    dropout : float
    max_len : int
        Maximum sequence length supported (default 5000).

    References
    ----------
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Section 3.5.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

# ----- Embeddings, verbatim from transvae/trans_models.py -----
class Embeddings(nn.Module):
    """Learnable token embeddings scaled by ``sqrt(d_model)``.

    The scaling factor prevents the embedding magnitudes from being
    overwhelmed by the additive positional encoding, following Vaswani et al.

    Parameters
    ----------
    d_model : int
        Embedding dimensionality.
    vocab : int
        Vocabulary size.

    References
    ----------
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Section 3.4.
    """
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# ----- WAEEncoder, verbatim from transvae/wae_models.py -----
class WAEEncoder(nn.Module):
    """GRU encoder with a deterministic linear bottleneck.

    Unlike VAE encoders, the WAE encoder uses a deterministic mapping
    ``z = W h[-1] + b`` with no stochastic reparameterisation.  Regularisation
    of the latent distribution comes from the MMD loss rather than a KL term.
    ``mu = 0`` and ``logvar = 0`` are returned as placeholders for API
    compatibility with the VAE training loop.

    Parameters
    ----------
    size : int
        GRU hidden size (``d_model``).
    d_latent : int
    N : int
        Number of GRU layers.
    dropout : float
    bypass_bottleneck : bool
    device : torch.device
    """
    def __init__(self, size, d_latent, N, dropout, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.norm = LayerNorm(size)
        self.linear_bypass = nn.Linear(size, d_latent)

    def forward(self, x, src_mask=None):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)          # (T, B, d)

        # Pack sequences so GRU skips PAD timesteps.
        # With packing, h[-1] for each sample is the hidden state at its
        # last *valid* timestep, not at the last (PAD) position.
        if src_mask is not None:
            if src_mask.dim() == 3:
                mask_2d = src_mask.squeeze(1).bool()
            elif src_mask.dim() == 2:
                mask_2d = src_mask.bool()
            else:
                mask_2d = None
            if mask_2d is not None:
                lengths = mask_2d.sum(dim=1).cpu().clamp(min=1)
                x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                    x, lengths, enforce_sorted=False
                )
                _, h = self.gru(x_packed, h)
            else:
                x, h = self.gru(x, h)
        else:
            x, h = self.gru(x, h)

        mem = self.norm(h[-1, :, :])
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
            mem = self.linear_bypass(mem)
        else:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
            mem = self.linear_bypass(mem)
        return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)

# ----- PropertyPredictor, verbatim from transvae/trans_models.py -----
class PropertyPredictor(nn.Module):
    """Optional property prediction head attached to the latent representation.

    Two modes are supported:

    * ``'deep_net'``: MLP with ReLU activations, outputting ``n_props`` values.
    * ``'decision_tree'``: Sklearn ``DecisionTreeClassifier`` baseline
      (non-differentiable; fitted inside ``forward()``; single output only).

    The head can predict either one property or several at once:

    * The original single-property head (``n_props=1``, ``return_logits=False``)
      predicts one antimicrobial probability. It applies a final sigmoid inside
      ``forward`` and is preserved exactly, so the shipped checkpoints load and
      behave unchanged.
    * A multi-property head (``n_props>1``, or ``return_logits=True``) predicts a
      vector of ``n_props`` raw values -- logits for binary properties and
      real-valued predictions for continuous ones. It does *not* apply any final
      activation; the loss applies the right one per column (sigmoid-based cross
      entropy for binary columns, plain regression for continuous columns). Keeping
      the head's output raw is what lets one head mix classification and
      regression targets.

    Parameters
    ----------
    d_pp : int
        Hidden layer width.
    depth_pp : int
        Number of linear layers (MLP depth) or ``max_depth`` (decision tree).
    d_latent : int
        Input dimensionality (latent size).
    type_pp : str
        ``'deep_net'`` or ``'decision_tree'``.
    n_props : int, optional
        Number of property outputs. Defaults to 1 (the legacy single-property
        head).
    return_logits : bool, optional
        If True, ``forward`` returns raw outputs with no final sigmoid (used for
        multi-property heads). If False (the default), the single output is passed
        through a sigmoid, exactly as the original head did.
    """
    def __init__(self, d_pp, depth_pp, d_latent, type_pp, n_props=1, return_logits=False):
        super().__init__()
        self.type_pp = type_pp
        self.n_props = n_props
        self.return_logits = return_logits

        if "decision_tree" in self.type_pp:
            assert n_props == 1, "decision_tree property head supports one property only"
            from sklearn.tree import DecisionTreeClassifier
            self.decision_tree = DecisionTreeClassifier(max_depth=depth_pp)

        else:
            prediction_layers = []
            for i in range(depth_pp):
                if i == 0:
                    linear_layer = nn.Linear(d_latent, d_pp)
                elif i == depth_pp - 1:
                    # Final layer produces one output per property. With n_props=1
                    # this is Linear(d_pp, 1) -- identical to the original head.
                    linear_layer = nn.Linear(d_pp, n_props)
                else:
                    linear_layer = nn.Linear(d_pp, d_pp)
                prediction_layers.append(linear_layer)
            self.prediction_layers = ListModule(*prediction_layers)

    def forward(self, x, true_prop):
        if "decision_tree" in self.type_pp:
            print(x.shape, true_prop.shape)
            self.decision_tree.fit(x.detach().numpy(), true_prop)
            print("N_classes:",self.decision_tree.n_classes_)
            x = self.decision_tree.predict_proba(x.detach().numpy()) #score on training data
        else:
            for idx, prediction_layer in enumerate(self.prediction_layers):
                if idx == len(self.prediction_layers) - 1:
                    x = prediction_layer(x)
                    # Legacy single-property head keeps its final sigmoid so old
                    # checkpoints behave exactly as before; the multi-property head
                    # returns raw values and lets the loss apply per-column
                    # activations.
                    if not self.return_logits:
                        x = torch.sigmoid(x)
                else:
                    x = torch.relu(prediction_layer(x))
        return x


# ===========================================================================
# Handoff wrapper. Written for this package; not part of the training library.
# ===========================================================================


class PeptideEncoder(nn.Module):
    """Tokenizer, encoder and AMP head for one trained cell.

    The submodules are built exactly as the training code built them, so the
    checkpoint's own parameter names load unchanged and the outputs are identical
    to the full model's.
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.architecture = str(config["architecture"])
        self.char_dict = config["char_dict"]
        self.src_len = int(config["src_len"])
        self.max_residues = int(config["max_residues"])
        self.d_latent = int(config["d_latent"])

        d_model = int(config["d_model"])
        vocab = len(self.char_dict)
        dropout = float(config["dropout"])

        if self.architecture == "transvae":
            # Mirrors TransVAE.build_model(): a prototype attention layer and
            # feed-forward block are deep-copied into the encoder layer.
            attn = MultiHeadedAttention(int(config["h"]), d_model)
            ff = PositionwiseFeedForward(d_model, int(config["d_ff"]), dropout)
            pos = PositionalEncoding(d_model, dropout)
            self.encoder = VAEEncoder(
                EncoderLayer(d_model, self.src_len, copy.deepcopy(attn),
                             copy.deepcopy(ff), dropout),
                int(config["N"]), self.d_latent,
                bool(config["bypass_bottleneck"]), float(config["eps_scale"]),
                target_conv_len=config["target_conv_len"])
            self.src_embed = nn.Sequential(Embeddings(d_model, vocab),
                                           copy.deepcopy(pos))
        elif self.architecture == "wae":
            # Mirrors WAE.build_model(). The WAE has no positional encoding.
            self.encoder = WAEEncoder(d_model, self.d_latent, int(config["N"]),
                                      dropout, bool(config["bypass_bottleneck"]),
                                      torch.device("cpu"))
            self.src_embed = Embeddings(d_model, vocab)
        else:
            raise ValueError("unknown architecture {!r}".format(self.architecture))

        # type_pp is 'deep_net' for every shipped cell, so the decision-tree branch
        # of PropertyPredictor -- the only thing here that would import sklearn --
        # is never taken.
        self.property_predictor = PropertyPredictor(
            int(config["d_pp"]), int(config["depth_pp"]), self.d_latent,
            str(config["type_pp"]))

    @classmethod
    def load(cls, path, device="cpu"):
        """Load one exported encoder and put it in evaluation mode."""
        blob = torch.load(path, map_location=device, weights_only=False)
        model = cls(blob["config"])
        # Strict: every tensor this encoder needs is in the file, the fixed
        # positional table included. That table is a formula, but the formula was
        # evaluated on the training cluster and sin/cos there differ from other
        # machines in the last bits, so the trained bytes are shipped rather than
        # recomputed. See README.md, "Provenance".
        model.load_state_dict(blob["state_dict"], strict=True)
        model.to(device)
        # The WAE encoder keeps its own device attribute, which it uses to allocate
        # the GRU's initial hidden state. .to() does not update a plain attribute, so
        # without this a non-CPU device gives "expected all tensors on the same
        # device" from inside the GRU.
        resolved = next(model.parameters()).device
        if hasattr(model.encoder, "device"):
            model.encoder.device = resolved
        model.eval()        # turns off dropout; without it the output varies per call
        return model

    def describe(self):
        """One-line summary of which trained cell this is."""
        return ("{architecture} hVAE, d_latent={d_latent}, from {source_checkpoint} "
                "(epoch {source_epoch})".format(**self.config))

    def tokenize(self, sequences):
        """Validate sequences and convert them to a padded id tensor.

        Raises rather than silently truncating or substituting, which is the one
        intentional difference from the training-time tokenizer: training read a
        fixed corpus, while anything may arrive here.
        """
        pad = self.char_dict["_"]
        start = self.char_dict["<start>"]
        end = self.char_dict["<end>"]

        rows = []
        for i, raw in enumerate(sequences):
            seq = str(raw).strip().upper()
            if not seq:
                raise ValueError("sequence {} is empty".format(i))
            bad = sorted(set(seq) - CANONICAL)
            if bad:
                raise ValueError(
                    "sequence {} contains non-canonical residues {}; only the 20 "
                    "standard amino acids are supported (see README.md)"
                    .format(i, bad))
            if len(seq) > self.max_residues:
                raise ValueError(
                    "sequence {} has {} residues; the maximum is {}. Truncating "
                    "would change the embedding silently, so this is refused."
                    .format(i, len(seq), self.max_residues))
            if len(seq) < 2:
                print("warning: sequence {} has {} residue; the training corpus ran "
                      "2-50, so this is out of distribution".format(i, len(seq)),
                      file=sys.stderr)
            ids = [start] + [self.char_dict[c] for c in seq] + [end]
            ids += [pad] * (self.src_len - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)

    @torch.inference_mode()
    def encode(self, sequences, batch_size=512):
        """Encode sequences, returning ``(embeddings, amp_scores)`` as float64.

        ``embeddings`` has shape (n, d_latent); ``amp_scores`` has shape (n,) and is
        P(AMP) from the model's own property head, which read exactly these vectors
        during training.
        """
        device = next(self.parameters()).device
        token_ids = self.tokenize(sequences)
        pad = int(self.char_dict["_"])

        chunks_z, chunks_p = [], []
        for begin in range(0, len(token_ids), batch_size):
            batch = token_ids[begin:begin + batch_size].to(device)

            # True where the position holds a real residue. Not optional: without
            # it the GRU runs through the padding and the attention encoder feeds
            # PAD rows into the convolution, both silently.
            src_mask = (batch != pad).unsqueeze(-2)

            if self.architecture == "transvae":
                # Four return values, mask positional. Slot 0 is a reparameterized
                # SAMPLE even in eval mode, so the deterministic vector is mu.
                _code, mu, _logvar, _pred_len = self.encoder(
                    self.src_embed(batch), src_mask)
                z = mu
            else:
                # Three return values, mask by keyword. A WAE has no posterior: slot
                # 0 is the code and the mu/logvar slots hold [0.0] placeholders.
                code, _mu, _logvar = self.encoder(
                    self.src_embed(batch), src_mask=src_mask)
                z = code

            amp = self.property_predictor(z, None)
            chunks_z.append(z.detach().cpu().numpy().astype(np.float64))
            chunks_p.append(amp.detach().cpu().numpy().astype(np.float64).reshape(-1))

        return np.concatenate(chunks_z), np.concatenate(chunks_p)


def read_sequences(path):
    """Read FASTA, or one sequence per line. Returns ``(ids, sequences)``."""
    with open(path) as handle:
        lines = [ln.strip() for ln in handle if ln.strip()]
    if any(ln.startswith(">") for ln in lines):
        ids, seqs, current = [], [], []
        for ln in lines:
            if ln.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
                ids.append(ln[1:].strip() or "seq{}".format(len(ids)))
            else:
                current.append(ln)
        if current:
            seqs.append("".join(current))
        if len(ids) != len(seqs):
            raise ValueError("FASTA has a header with no sequence")
        return ids, seqs
    if lines and lines[0].lower() in {"peptides", "sequence", "sequences"}:
        lines = lines[1:]
    return ["seq{}".format(i) for i in range(len(lines))], lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True,
                    help="FASTA, or one sequence per line")
    ap.add_argument("--out", default="embeddings.npz", help="output .npz")
    ap.add_argument("--weights", default="encoders/hvae_wae_d064_encoder.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    ids, seqs = read_sequences(args.inp)
    model = PeptideEncoder.load(args.weights, device=args.device)
    z, amp = model.encode(seqs, batch_size=args.batch_size)
    np.savez(args.out, ids=np.array(ids), sequences=np.array(seqs),
             embeddings=z, amp_score=amp)
    print("{}\nencoded {} peptides -> {}  embeddings {}, amp_score {}"
          .format(model.describe(), len(seqs), args.out, z.shape, amp.shape))


if __name__ == "__main__":
    main()
