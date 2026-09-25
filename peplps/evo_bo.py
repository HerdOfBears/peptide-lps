"""EVO_BO_2026: mutation-space Bayesian optimization over sequences.

Implements the explorer strategy of Ma et al. (2026),
https://doi.org/10.34133/research.1420 (bo/explorers.py::BO_EVO), behind the
same ask/tell interface as MultiFidelityBO_Wu2019KG.

Key difference from the BoTorch class: the design is a *string*, not a row of
floats, so the acquisition function cannot be optimized by gradient ascent.
Instead it is maximized by enumeration over a stochastically generated pool of
mutants of the current incumbents. Every candidate is therefore a real,
synthesizable sequence and the GP is evaluated at its genuine embedding -- no
encoder inversion is needed and no decode/optimize mismatch can arise.

Data layout
-----------
train_x : (n, 2) object array. Column 0 is the sequence (str), column 1 is the
          fidelity (float). ``fidelity_col == 1``.
train_obj : (n, 1) float array, MAXIMIZED.

Candidates from ``suggest()`` carry their chosen fidelity in the last column so
the caller can dispatch to the right scorer, exactly as in the BoTorch class.

Single-fidelity use
-------------------
Default. ``params={"FIDELITIES": [1.0]}`` (the default) makes every candidate
target fidelity, makes the cost model constant, and short-circuits the
cost-weighting, so the loop reduces to ordinary single-fidelity BO.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

AAS = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def _seq_to_mut(seq: str, wt: str) -> str:
    """Human-readable mutation string, e.g. 'A3K,L7W' or 'WT'."""
    if len(seq) != len(wt):
        return seq
    mut = ",".join(
        f"{wt[i]}{i + 1}{seq[i]}" for i in range(len(wt)) if wt[i] != seq[i]
    )
    return mut if mut else "WT"


class _IdentityEncoder:
    """One-hot encoder used when the caller supplies no encoder.

    Exposes ``per_position_dim`` so ARD position weighting stays available.
    """

    name = "onehot"

    def __init__(self, alphabet: str = AAS):
        self.alphabet = alphabet
        self.per_position_dim = len(alphabet)
        self._index = {aa: i for i, aa in enumerate(alphabet)}

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        n_aa = len(self.alphabet)
        out = np.zeros((len(sequences), len(sequences[0]) * n_aa), dtype=float)
        for r, seq in enumerate(sequences):
            for pos, aa in enumerate(seq):
                idx = self._index.get(aa)
                if idx is not None:
                    out[r, pos * n_aa + idx] = 1.0
        return out

class _NResEncoder:
    """
    Count number of each residue in sequence, ignoring positional info.
    """
    name = "numresidues"

    def __init__(self, alphabet:str = AAS):
        self.alphabet = alphabet
        self.per_position_dim = len(alphabet)
        self._index = {aa: i for i, aa in enumerate(alphabet)}

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        n_aa = len(self.alphabet)
        out = np.zeros((len(sequences), n_aa), dtype=float)
        for r, seq in enumerate(sequences):
            for aa, idx in self._index.items():
                _n_aa = seq.count(aa)
                out[r, idx] = _n_aa
        return out

# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class EVO_BO_2026:
    """
    Implementation of a mutation Bayesian Optimization approach, inspired
    by the work of Ma et al. (2026) https://doi.org/10.34133/research.1420

    A Gaussian process regressor operates on a representation/embedding space
    determined by a provided encoder model (such as ESM2). No decoder is
    used, instead peptides are determined/searched entirely at the level of
    sequence through a evolutionary algorithm approach.

    Rough procedure:
    - initial training set
    - embed initial training set into feature space
    - fit GPR to the (feature, objective value) pairs
    - Optimize acquisition function by sampling batches of sequences
        - batches of sequences are obtained via mutating top candidate
        - allows controls of ~number of mutations, Hamming distance
    - acquisition function suggests a sequence to score.
    - Obtain score from scorer
    - update training set
    - repeat
    """

    LOW_FIDELITY = 0.0
    HIGH_FIDELITY = 1.0

    def __init__(self, train_x, train_obj, params=None):
        params = params or {}
        self.smoke_test = params.get("SMOKE_TEST", False)

        # --- data layout ----------------------------------------------------
        # design is a string; column 1 of train_x holds the fidelity.
        self.fidelity_col = 1
        self.dim = 2

        # --- fidelities & cost ---------------------------------------------
        # Single fidelity by default -> [1.0] -> cost-weighting is a no-op.
        self.fidelities = [float(f) for f in params.get("FIDELITIES", [self.HIGH_FIDELITY])]
        if self.HIGH_FIDELITY not in self.fidelities:
            raise ValueError("FIDELITIES must include the target fidelity 1.0")
        self.target_fidelity = self.HIGH_FIDELITY
        self.is_multifidelity = len(self.fidelities) > 1
        # cost(s) = fixed_cost + weight * s  ->  cost(0)=1, cost(1)=10
        self.cost_fixed  = float(params.get("COST_FIXED", 1.0))
        self.cost_weight = float(params.get("COST_WEIGHT", 9.0))
        # Guards against the cost-weighted acquisition collapsing onto the
        # cheapest fidelity and never validating anything.
        self.min_high_fidelity_per_batch = int(
            params.get("MIN_HIGH_FIDELITY_PER_BATCH", 1 if self.is_multifidelity else 0)
        )

        # --- search / proposal budgets --------------------------------------
        self.batch_size = params.get("BATCH_SIZE", 4)
        if self.batch_size==1:
            self.min_high_fidelity_per_batch=0

        # analogue of model_queries_per_round / expmt_queries_per_round
        self.candidates_per_query = params.get(
            "CANDIDATES_PER_QUERY", 256 if not self.smoke_test else 16
        )
        self.n_seeds = params.get("N_SEEDS", 4 if not self.smoke_test else 2)
        self.min_mutations = max(1, int(params.get("MIN_MUTATIONS", 1)))
        self.max_mutations = params.get("MAX_MUTATIONS", 3)
        self.mutation_rate = float(params.get("MUTATION_RATE", 1.0))  # Poisson lambda
        self.batch_diversity_min_hamming = int(
            params.get("BATCH_DIVERSITY_MIN_HAMMING", 2)
        )
        self.alphabet = params.get("ALPHABET", AAS)

        # --- acquisition -----------------------------------------------------
        self.acq_name = params.get("ACQ", "EI").upper()
        # NOTE: the reference implementation defaults to UCB with beta=0.2,
        # which is close to pure greedy exploitation. Default here is EI; if
        # you pick UCB, prefer beta ~2.0.
        self.beta = float(params.get("BETA", 2.0))
        self.xi   = float(params.get("XI", 0.01))  # EI/PI exploration offset
        self.synthesis_penalty_weight = float(
            params.get("SYNTHESIS_PENALTY_WEIGHT", 0.0)
        )
        self.synthesis_fn = params.get("SYNTHESIS_FN", None)  # seq -> penalty
        self.max_synthesis_penalty = params.get("MAX_SYNTHESIS_PENALTY", None)

        # --- encoder ---------------------------------------------------------
        self.encoder = params.get("ENCODER", None) or _IdentityEncoder(self.alphabet)
        if not hasattr(self.encoder, "encode"):
            raise TypeError("ENCODER must expose .encode(list[str]) -> ndarray")
        self.per_position_dim = getattr(self.encoder, "per_position_dim", None)
        self.position_sampling = params.get(
            "POSITION_SAMPLING", "ard" if self.per_position_dim else "uniform"
        )

        self.rng = np.random.default_rng(params.get("SEED", 0))
        self._embed_cache: dict[str, np.ndarray] = {}

        # --- dataset is owned by this object --------------------------------
        self.train_x = self._as_design_array(train_x)
        self.train_obj = np.asarray(train_obj, dtype=float).reshape(-1, 1)
        if len(self.train_x) != len(self.train_obj):
            raise ValueError("train_x and train_obj must have the same length")
        if len(self.train_x) == 0:
            raise ValueError("need at least one initial observation")

        self.seq_len = len(self.train_x[0, 0])
        self.wt_sequence = str(self.train_x[int(np.argmax(self.train_obj)), 0])
        if self.max_mutations is None:
            self.max_mutations = min(3, self.seq_len)
        self.max_mutations = int(min(self.max_mutations, self.seq_len))
        if self.max_mutations < self.min_mutations:
            raise ValueError("MAX_MUTATIONS must be >= MIN_MUTATIONS")
        if self.batch_diversity_min_hamming > 2 * self.max_mutations:
            raise ValueError(
                "BATCH_DIVERSITY_MIN_HAMMING is too strict for the mutation radius"
            )
        self._check_capacity()

        self._fit_model()

    # ------------------------------------------------------------ data utils
    def _as_design_array(self, x) -> np.ndarray:
        """Coerce input to an (n, 2) object array [sequence, fidelity]."""
        if isinstance(x, np.ndarray) and x.dtype == object and x.ndim == 2:
            out = x.copy()
        else:
            rows = []
            for item in x:
                if isinstance(item, str):
                    rows.append([item, self.target_fidelity])
                else:
                    seq, fid = item[0], float(item[1])
                    rows.append([str(seq), fid])
            out = np.empty((len(rows), 2), dtype=object)
            for i, (s, f) in enumerate(rows):
                out[i, 0], out[i, 1] = str(s), float(f)
        for f in out[:, 1]:
            if float(f) not in self.fidelities:
                raise ValueError(f"unknown fidelity {f}; expected one of {self.fidelities}")
        return out

    def _check_capacity(self):
        """Mutation neighbourhood must be able to supply the requested pool."""
        aa_options = max(0, len(self.alphabet) - 1)
        capacity = sum(
            math.comb(self.seq_len, k) * (aa_options ** k)
            for k in range(self.min_mutations, self.max_mutations + 1)
            if k <= self.seq_len
        )
        need = max(self.batch_size * 8, self.candidates_per_query)
        if capacity < need:
            raise ValueError(
                f"mutation neighbourhood holds only {capacity} sequences; need "
                f"~{need}. Increase MAX_MUTATIONS or lower CANDIDATES_PER_QUERY."
            )

    def _encode(self, sequences: Sequence[str]) -> np.ndarray:
        """Encode with a cache; embeddings are the expensive part."""
        missing = [s for s in sequences if s not in self._embed_cache]
        if missing:
            uniq = list(dict.fromkeys(missing))
            embedded = np.asarray(self.encoder.encode(uniq), dtype=float)
            if embedded.ndim == 1:
                embedded = embedded.reshape(len(uniq), -1)
            for s, vec in zip(uniq, embedded):
                self._embed_cache[s] = vec
        return np.vstack([self._embed_cache[s] for s in sequences])

    def _features(self, sequences: Sequence[str], fidelities: Sequence[float]) -> np.ndarray:
        """GP input: embedding with the fidelity appended as one extra column.

        The extra column gets its own ARD length scale, so the GP learns how
        strongly the fidelities are correlated rather than being told.
        """
        emb = self._encode(list(sequences))
        if not self.is_multifidelity:
            return emb
        fid = np.asarray(fidelities, dtype=float).reshape(-1, 1)
        return np.hstack([emb, fid])

    def cost(self, fidelity: float) -> float:
        return self.cost_fixed + self.cost_weight * float(fidelity)

    # ---------------------------------------------------------------- model
    def _fit_model(self):
        """(Re)build and fit the surrogate on the current dataset."""
        X = self._features(list(self.train_x[:, 0]), list(self.train_x[:, 1]))
        y = self.train_obj.ravel()

        n_dims = X.shape[1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(n_dims),          # ARD: one length scale per dim
            length_scale_bounds=(1e-2, 1e4),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,          # standardize targets; predictions come back
            n_restarts_optimizer=0 if self.smoke_test else 4,
            random_state=int(self.rng.integers(2**31 - 1)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)

    @property
    def ard_lengthscale(self) -> Optional[np.ndarray]:
        """Fitted per-dimension length scales, embedding dims only."""
        try:
            k = self.model.kernel_
            matern = None
            stack = [k]
            while stack:
                node = stack.pop()
                if isinstance(node, Matern):
                    matern = node
                    break
                for attr in ("k1", "k2"):
                    if hasattr(node, attr):
                        stack.append(getattr(node, attr))
            if matern is None:
                return None
            ls = np.atleast_1d(np.asarray(matern.length_scale, dtype=float))
            if self.is_multifidelity and ls.size > 1:
                ls = ls[:-1]           # drop the fidelity column's length scale
            return ls
        except Exception:
            return None

    def _predict(self, sequences: Sequence[str], fidelity: float):
        X = self._features(sequences, [fidelity] * len(sequences))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = self.model.predict(X, return_std=True)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), 1e-12)

    # ----------------------------------------------------- proposal sampling
    def _position_weights(self) -> np.ndarray:
        """Per-position mutation weights from the GP's ARD length scales.

        Short length scale == the GP found that position predictive == mutate
        it more often. Falls back to uniform when the encoder is pooled (no
        per-position structure to map relevance back onto).
        """
        uniform = np.full(self.seq_len, 1.0 / self.seq_len)
        if self.position_sampling == "uniform" or not self.per_position_dim:
            return uniform
        ls = self.ard_lengthscale
        if ls is None or ls.size != self.seq_len * int(self.per_position_dim):
            return uniform
        ls_per_pos = ls.reshape(self.seq_len, int(self.per_position_dim))
        relevance = 1.0 / (ls_per_pos.mean(axis=1) + 1e-8)
        total = relevance.sum()
        if not np.isfinite(total) or total <= 0:
            return uniform
        return relevance / total

    def _amino_acid_sampler(self, current_aa:str, method:str="uniform")->str:
        """
        samples from amino acid choices using 'method' and
        excluding 'current_aa' from the choices
        """
        # get AAs we can mutate to
        choices = [aa for aa in self.alphabet if aa != current_aa]

        if method=="uniform":
            # sample amino acid uniformly from choices
            _idx = int(self.rng.integers(len(choices)))

        new_aa = choices[_idx]
        return new_aa

    def _random_mutant(self, seed_seq: str, pos_weights: np.ndarray) -> str:
        seq = list(seed_seq)

        # get number of mutations to perform for given sequence
        n_mut = int(self.rng.poisson(self.mutation_rate))
        _max_mut = min(n_mut, self.max_mutations, self.seq_len)
        n_mut = max(self.min_mutations, _max_mut)

        # get positions at which the mutations will occur
        # all unique positions.
        positions = self.rng.choice(
            self.seq_len, size=n_mut, replace=False, p=pos_weights
        )

        # perform mutations with 'method' sampling over amino acids
        # a mutation must be different from current AA
        for pos in positions:
            _current_aa = seq[pos]

            # get mutation
            _new_aa = self._amino_acid_sampler(_current_aa)

            seq[pos] = _new_aa

        return "".join(seq)

    def _seed_sequences(self) -> list[str]:
        """Incumbents to mutate from: best observed at the target fidelity.

        Thompson-style: draw from the GP posterior at measured target-fidelity
        points and keep the top draws, so the seeds vary between rounds.
        """
        at_target = self.train_x[:, 1].astype(float) == self.target_fidelity
        pool = (
            list(self.train_x[at_target, 0])
            if at_target.any()
            else list(self.train_x[:, 0])
        )
        pool = list(dict.fromkeys(pool))
        mu, sd = self._predict(pool, self.target_fidelity)
        draws = self.rng.normal(mu, sd)
        order = np.argsort(draws)[::-1]
        k = min(self.n_seeds, len(pool))
        return [pool[i] for i in order[:k]]

    def _generate_pool(self, measured: set[str]) -> list[str]:
        """Sample the mutation neighbourhood of the incumbents.

        This is the acquisition-function optimizer: no gradients are available
        over strings, so alpha is maximized by enumeration over this pool.
        """
        pos_weights = self._position_weights()
        seeds = self._seed_sequences()
        pool: list[str] = []
        seen: set[str] = set()
        max_draws = self.candidates_per_query * 20
        draws = 0
        while len(pool) < self.candidates_per_query and draws < max_draws:
            draws += 1
            seed_seq = seeds[int(self.rng.integers(len(seeds)))]
            cand = self._random_mutant(seed_seq, pos_weights)
            if cand in seen or cand in measured:
                continue
            if self.max_synthesis_penalty is not None and self.synthesis_fn is not None:
                if float(self.synthesis_fn(cand)) > self.max_synthesis_penalty:
                    continue
            seen.add(cand)
            pool.append(cand)
        if not pool:
            raise RuntimeError(
                "no novel feasible candidates were generated; relax the "
                "synthesis filter or widen the mutation radius"
            )
        return pool

    # ------------------------------------------------------------ acquisition
    def _acquisition(self, mu: np.ndarray, sd: np.ndarray, best_f: float) -> np.ndarray:
        name = self.acq_name
        if name == "UCB":
            return mu + self.beta * sd
        if name == "GREEDY":
            return mu
        if name in ("EI", "PI"):
            imp = mu - best_f - self.xi
            z = imp / sd
            if name == "PI":
                return norm.cdf(z)
            return imp * norm.cdf(z) + sd * norm.pdf(z)
        if name == "TS":
            return self.rng.normal(mu, sd)
        raise ValueError(f"unsupported acquisition: {self.acq_name}")

    def _incumbent(self) -> float:
        """Best MEASURED value at the target fidelity.

        Deliberately not the best *predicted* value: using predictions inflates
        the incumbent and makes EI/PI silently over-conservative.
        """
        at_target = self.train_x[:, 1].astype(float) == self.target_fidelity
        if at_target.any():
            return float(self.train_obj[at_target].max())
        return float(self.train_obj.max())

    # ------------------------------------------------------------------- ask
    def suggest(self):
        """Propose the next batch of points to evaluate (NO scoring done here).

        Returns
        -------
        new_x : (q, 2) object array
            Column 0 is the sequence, column 1 is the fidelity (use it to pick
            which scoring function to call for each row).
        cost : float
            Total cost of evaluating this batch, per the affine cost model.
        """
        measured = set(self.train_x[:, 0])
        pool = self._generate_pool(measured)
        best_f = self._incumbent()

        # score every (candidate, fidelity) pair, cost-weighted
        rows = []
        for fid in self.fidelities:
            mu, sd = self._predict(pool, fid)
            acq = self._acquisition(mu, sd, best_f)
            if self.synthesis_penalty_weight and self.synthesis_fn is not None:
                pen = np.array([float(self.synthesis_fn(s)) for s in pool])
                acq = acq - self.synthesis_penalty_weight * pen
            if self.is_multifidelity:
                # Inverse-cost weighting is only meaningful for a NON-NEGATIVE,
                # improvement-style utility. EI/PI already are; UCB/TS/Greedy
                # are raw objective scales and can be negative, in which case
                # dividing by a larger cost makes them *larger* and the
                # weighting inverts. Rebase those onto improvement-over-
                # incumbent and clip at zero first.
                if self.acq_name in ("UCB", "TS", "GREEDY"):
                    acq = np.maximum(acq - best_f, 0.0)
                acq = acq / self.cost(fid)
            for i, seq in enumerate(pool):
                rows.append((float(acq[i]), seq, float(fid), float(mu[i])))

        rows.sort(key=lambda r: -r[0])

        # greedy batch selection under a Hamming-diversity constraint
        selected: list[tuple[str, float]] = []
        chosen_seqs: list[str] = []
        min_h = self.batch_diversity_min_hamming
        n_high = 0
        for _, seq, fid, _mu in rows:
            if len(selected) >= self.batch_size:
                break
            if any(s == seq and f == fid for s, f in selected):
                continue
            # same sequence at a *different* fidelity is allowed and useful;
            # only enforce diversity against distinct sequences.
            if seq not in chosen_seqs and min_h > 0:
                if any(_hamming(seq, s) < min_h for s in chosen_seqs):
                    continue
            selected.append((seq, fid))
            if seq not in chosen_seqs:
                chosen_seqs.append(seq)
            if fid == self.target_fidelity:
                n_high += 1

        # enforce a floor on target-fidelity evaluations
        if self.is_multifidelity and n_high < self.min_high_fidelity_per_batch:
            need = self.min_high_fidelity_per_batch - n_high
            for idx in range(len(selected) - 1, -1, -1):
                if need == 0:
                    break
                if selected[idx][1] != self.target_fidelity:
                    selected[idx] = (selected[idx][0], self.target_fidelity)
                    need -= 1

        if not selected:
            raise RuntimeError("no candidate satisfied the batch diversity constraint")

        new_x = np.empty((len(selected), 2), dtype=object)
        for i, (seq, fid) in enumerate(selected):
            new_x[i, 0], new_x[i, 1] = seq, fid
        total_cost = float(sum(self.cost(f) for _, f in selected))
        return new_x, total_cost

    # ------------------------------------------------------------------ tell
    def register_observations(self, new_x, new_obj):
        """Add externally-computed scores to the dataset and refit the model.

        Parameters
        ----------
        new_x : (q, 2) object array
            The candidates returned by suggest() (fidelity column included).
        new_obj : (q, 1) array
            Observed objective values, in the same row order as new_x.
        """
        new_x = self._as_design_array(new_x)
        new_obj = np.asarray(new_obj, dtype=float).reshape(-1, 1)
        if len(new_x) != len(new_obj):
            raise ValueError("new_x and new_obj must have the same length")
        keep = np.isfinite(new_obj).ravel()
        if not keep.all():
            new_x, new_obj = new_x[keep], new_obj[keep]
        if len(new_x) == 0:
            return
        self.train_x = np.vstack([self.train_x, new_x])
        self.train_obj = np.vstack([self.train_obj, new_obj])
        self._fit_model()

    # ---------------------------------------------------------- recommendation
    def get_recommendation(self):
        """Best design at the target fidelity (maximizer of the posterior mean).

        Restricted to sequences that have actually been evaluated at the target
        fidelity, so the recommendation is always something observed rather
        than an extrapolation.
        """
        at_target = self.train_x[:, 1].astype(float) == self.target_fidelity
        pool = list(self.train_x[at_target, 0]) if at_target.any() else list(self.train_x[:, 0])
        pool = list(dict.fromkeys(pool))
        mu, _ = self._predict(pool, self.target_fidelity)
        best = pool[int(np.argmax(mu))]
        out = np.empty((1, 2), dtype=object)
        out[0, 0], out[0, 1] = best, self.target_fidelity
        return out

    # ------------------------------------------------------------------ misc
    def describe(self, seq: str) -> str:
        """Mutation string relative to the initial best sequence."""
        return _seq_to_mut(seq, self.wt_sequence)