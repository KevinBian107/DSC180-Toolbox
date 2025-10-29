# CLAUDE_ML.md

This file provides comprehensive guidance for Claude when working with JAX-based machine learning code for robotics, embodied AI, and MuJoCo simulations.

## Core Development Philosophy

### KISS (Keep It Simple, Stupid)

Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)

Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

## Core ML Philosophy

### Reproducibility First
- **Always set random seeds** at the start of experiments
- **Log all hyperparameters** to experiment tracking (W&B, MLflow)
- **Version control data paths** and preprocessing steps
- **Save exact environment specs** (package versions, hardware info)

### Fail Fast, Debug Faster
- **Add assertions** for tensor shapes at critical points
- **Use `jax.debug.print`** for runtime inspection without breaking JIT
- **Validate data immediately** after loading
- **Check for NaNs/Infs** early and often

### Modular Experimentation
- **Separate data, model, training, and inference** into distinct modules
- **Use configuration files** (YAML/JSON) for all experiments
- **Registry pattern** for models and datasets
- **Never hardcode paths** - use Path objects and config

## 🏗️ Project Structure for ML

```
ncssm/                              # Root project directory
├── configs/                         # YAML configuration files
│   ├── s5_sa.yaml                  # S5 state→action config
│   ├── s5_si.yaml                  # S5 state→intention config
│   ├── s5_sic.yaml                 # S5 state→intention→control config
│   ├── s5_ssi.yaml                 # S5 syllable state→intention config
│   ├── mamba.yaml                  # Mamba baseline config
│   ├── rollout.yaml                # Rollout/evaluation config
│   └── kpms_chunking.yaml          # Keypoint matching config
│
├── dataloaders/                     # Data loading and preprocessing
│   ├── __init__.py                 # Dataset registry
│   ├── load_h5.py                  # HDF5 data loader (main)
│   └── rollout_training_data.py    # Rollout data generation
│
├── trainer/                         # Training modules
│   ├── deep_ssm/                   # Deep state space models
│   │   ├── models/                 # Model architectures
│   │   │   ├── s5.py              # S5 layer implementation
│   │   │   ├── minimamba.py       # Mamba layer implementation
│   │   │   └── seq_models.py      # Full sequence models
│   │   ├── s5_trainer.py          # S5 training script
│   │   ├── s5_controller_trainer.py  # S5 controller training
│   │   ├── s5_syllable_trainer.py    # S5 syllable training
│   │   └── mamba_trainer.py       # Mamba training script
│   │
│   ├── prob_ssm/                   # Probabilistic SSMs (AR-HMM, etc.)
│   │   └── arhmm_trainer.py
│   │
│   └── train_helper.py             # Training utilities (TrainStateWithStats, etc.)
│
├── inference/                       # Inference and evaluation
│   ├── deep_ssm/                   # Deep SSM inference
│   │   ├── inferencer.py          # Model predictor classes (S5Predictor, etc.)
│   │   └── ssm_rollout_state_action.py  # Rollout evaluation
│   │
│   ├── prob_ssm/                   # Probabilistic SSM inference
│   │   └── inferencer.py
│   │
│   ├── inference_helper.py         # Inference utilities
│   └── render.py                   # Visualization/rendering
│
├── segmentation/                    # Behavioral segmentation
│   ├── chunk.py                    # Chunking algorithms
│   ├── fit_eval.py                 # Fitting and evaluation
│   └── hypertune.py                # Hyperparameter tuning
│
├── docs/                           # Documentation
├── notebooks/                       # Jupyter notebooks for analysis
├── slurm/                          # SLURM job scripts
├── tests/                          # Unit tests
├── xml/                            # MuJoCo XML models
│
├── train.py                        # Main training entry point
├── rollout.py                      # Main rollout entry point
└── pyproject.toml                  # UV/pip dependencies
```

**Key Directory Conventions:**
- **configs/**: All YAML files for different model modes and experiments
- **dataloaders/**: HDF5 data loading with multiprocessing support
- **trainer/**: Training scripts organized by model family (deep_ssm, prob_ssm)
- **inference/**: Predictor classes and rollout evaluation
- **segmentation/**: Behavioral chunking and analysis tools

## 🎯 JAX Best Practices

### JIT Compilation

```python
import jax
import jax.numpy as jp

# ✅ DO: JIT pure functions with clear signatures
@jax.jit
def train_step(
    state: TrainState,
    batch: dict[str, jp.ndarray]
) -> tuple[TrainState, dict[str, float]]:
    """Single training step with gradient update."""
    
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch["inputs"])
        loss = jp.mean((preds - batch["targets"]) ** 2)
        return loss, {"mse": loss}
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


# ❌ DON'T: Put I/O or stateful operations inside JIT
@jax.jit
def bad_train_step(state, batch):
    # This will fail or cause issues
    print(f"Loss: {loss}")  # ❌ Side effects
    wandb.log({"loss": loss})  # ❌ I/O inside JIT
    return state


# ✅ DO: Handle I/O outside JIT
def train_epoch(state, dataloader):
    for batch in dataloader:
        state, metrics = train_step(state, batch)  # JITed
        wandb.log(metrics)  # Outside JIT
    return state
```

### Debugging JAX Code

```python
import jax
import jax.numpy as jp

# ✅ Use jax.debug.print for runtime debugging
@jax.jit
def debug_forward(params, x):
    jax.debug.print("Input shape: {}", x.shape)
    jax.debug.print("Input mean: {}", jp.mean(x))
    
    y = model(params, x)
    
    jax.debug.print("Output shape: {}", y.shape)
    jax.debug.print("Has NaN: {}", jp.any(jp.isnan(y)))
    
    return y


# ✅ Use jax.debug.breakpoint for interactive debugging
@jax.jit
def conditional_debug(x):
    jax.lax.cond(
        jp.any(jp.isnan(x)),
        lambda: jax.debug.breakpoint(),  # Pause if NaN detected
        lambda: None
    )
    return process(x)


# ✅ Check shapes with assertions (in non-JIT code)
def validate_batch(batch: dict[str, jp.ndarray]):
    """Validate batch shapes before training."""
    B, T, D = batch["states"].shape
    assert batch["actions"].shape == (B, T, 15), \
        f"Expected actions (B, T, 15), got {batch['actions'].shape}"
    assert not jp.any(jp.isnan(batch["states"])), \
        "NaN detected in states"
```

### Memory Management

```python
# ✅ Clear JAX cache periodically for long training runs
import jax

def train_with_cache_clearing(state, dataloader, clear_every=100):
    for i, batch in enumerate(dataloader):
        state, metrics = train_step(state, batch)
        
        if i % clear_every == 0:
            jax.clear_caches()  # Prevent memory buildup
    
    return state


# ✅ Use gradient checkpointing for large models
from flax import linen as nn

class CheckpointedModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Use remat for memory-intensive blocks
        @nn.remat
        def expensive_block(x):
            return heavy_computation(x)
        
        return expensive_block(x)
```

## 📊 Data Pipeline Standards

### HDF5 Data Structure

Your project uses HDF5 files with the following structure:

```
data/
├── clip_0.h5
├── clip_1.h5
├── ...
└── clip_841.h5

Each clip_i.h5 contains:
├── /obs          # Full observations (T, 537) - includes ref_obs + qpos + qvel + etc.
├── /ctrl         # Control actions (T, 15)
├── /latent_mean  # Intention vectors from VAE (T, latent_dim)
└── /syllables    # Discrete behavior labels (T,) [optional]
```

**Critical HDF5 Loading Pattern:**

```python
import h5py
import numpy as np
from pathlib import Path

def _load_from_h5py(file: h5py.File, group_path: str = "/"):
    """
    Recursively load pytree structure from HDF5.
    
    Handles:
    - Datasets → numpy arrays
    - Groups with numeric keys → lists (sorted by int)
    - Groups with string keys → dicts
    """
    group = file[group_path]
    
    if isinstance(group, h5py.Dataset):
        return group[()]  # Load dataset to numpy
    
    if isinstance(group, h5py.Group):
        # List-like: all keys are digits
        if all(k.isdigit() for k in group):
            return [_load_from_h5py(file, f"{group_path}/{k}")
                    for k in sorted(group, key=int)]
        # Dict-like: string keys
        return {k: _load_from_h5py(file, f"{group_path}/{k}") 
                for k in group}
    
    raise TypeError(f"Unsupported HDF5 node: {type(group)}")


def load_clip_data(
    root_dir: str,
    clip_idx: int,
    data_key: str = "obs"
) -> np.ndarray:
    """
    Load single clip data.
    
    Args:
        root_dir: Directory with clip_i.h5 files
        clip_idx: Clip index (0-841)
        data_key: HDF5 key ("obs", "ctrl", "latent_mean", "syllables")
    
    Returns:
        numpy array of shape (T, D)
    """
    path = Path(root_dir) / f"clip_{clip_idx}.h5"
    
    if not path.exists():
        raise FileNotFoundError(f"Clip file not found: {path}")
    
    with h5py.File(path, "r") as f:
        data = _load_from_h5py(f, f"/{data_key}")
    
    return data.astype(np.float32)
```

### Dataset Loading

```python
from typing import Iterator, Dict, Tuple
import jax.numpy as jp
import h5py, os, random
import numpy as np
import multiprocessing as mp
from functools import partial
from pathlib import Path

class RoboticsDataset:
    """
    Standard dataset interface for robotics HDF5 data.
    
    **Modeling Modes (mmode):**
    - "state_action": s_t → a_{t+1} (action prediction)
    - "pure_state": s_t → s_{t+1} (state prediction)
    - "state_intention": s_t → i_{t+1} (intention prediction)
    - "state_intention_controller": s_t → i_t → a_{t+1} (hierarchical control)
    - "syllable_state_intention": (y_t, s_t) → a_{t+1} (syllable-conditioned)
    - "pure_intention": i_t → i_{t+1} (intention dynamics)
    
    Args:
        root_dir: Path to directory with clip_i.h5 files
        batch_size: Total batch size (will be split by train_frac)
        split: 'train' or 'val'
        train_frac: Fraction of data for training
        seg_len: Sequence length to extract (default 490)
        preload: Load all clips into RAM (faster but memory-intensive)
        test_batch: Use single clip for overfitting test
        mmode: Modeling mode (see above)
        noise_std: Gaussian noise std for train augmentation
        clip_idx: Specific clip index for test_batch
        seed: Random seed
    """
    
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        split: str = "train",
        train_frac: float = 0.9,
        seg_len: int = 490,
        preload: bool = False,
        test_batch: bool = False,
        mmode: str = "state_action",
        noise_std: float = 0.01,
        clip_idx: int = 25,
        seed: int = 0,
        **kwargs
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = split
        self.seg_len = seg_len
        self.mmode = mmode
        self.noise_std = noise_std if split == "train" else 0.0
        self.test_batch = test_batch
        self.clip_idx = clip_idx
        
        # Validate
        assert self.root_dir.exists(), f"Data path not found: {self.root_dir}"
        assert split in ("train", "val"), f"Invalid split: {split}"
        assert 0 < train_frac < 1, f"train_frac must be in (0,1), got {train_frac}"
        
        # Get all clip IDs
        self.clip_ids = sorted([
            int(f.split("_")[1].split(".")[0])
            for f in os.listdir(self.root_dir)
            if f.startswith("clip_") and f.endswith(".h5")
        ])
        
        print(f"Found {len(self.clip_ids)} clips in {self.root_dir}")
        
        # Split train/val
        if not test_batch:
            rng = random.Random(seed)
            rng.shuffle(self.clip_ids)
        
        split_idx = int(train_frac * len(self.clip_ids))
        self.clip_ids = self.clip_ids[:split_idx] if split == "train" else self.clip_ids[split_idx:]
        
        # Adjust batch size for split
        self.batch_size = max(1, int(len(self.clip_ids) * (train_frac if split == "train" else (1 - train_frac))))
        self.batch_size = min(self.batch_size, batch_size)
        
        print(f"{split} using {len(self.clip_ids)} clips, batch_size={self.batch_size}")
        
        # Preload if requested
        if preload:
            self._preload_clips()
        
        self.rng = np.random.default_rng(seed)
    
    def _preload_clips(self):
        """Load all clips into memory using multiprocessing."""
        print(f"Preloading {len(self.clip_ids)} clips...")
        
        with mp.Pool(mp.cpu_count()) as pool:
            # Load states
            self.states = pool.map(
                partial(load_clip_data, self.root_dir, data_key="obs"),
                self.clip_ids
            )
            
            # Load actions or intentions based on mmode
            if self.mmode in ("state_action", "state_intention_controller", "syllable_state_intention"):
                data_key = "ctrl"
            elif self.mmode == "state_intention":
                data_key = "latent_mean"
            elif self.mmode == "pure_state":
                data_key = "obs"
            elif self.mmode == "pure_intention":
                data_key = "latent_mean"
            else:
                raise ValueError(f"Unknown mmode: {self.mmode}")
            
            self.actions = pool.map(
                partial(load_clip_data, self.root_dir, data_key=data_key),
                self.clip_ids
            )
        
        self.states = np.stack(self.states)
        self.actions = np.stack(self.actions)
        
        print(f"✓ Preloaded: states {self.states.shape}, actions {self.actions.shape}")
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[jp.ndarray, jp.ndarray]:
        """
        Generate next batch.
        
        Returns:
            (x_in, x_tgt) where:
            - x_in: Input states (B, T, state_dim)
            - x_tgt: Target actions/states (B, T, action_dim)
        """
        # Select clips
        if self.test_batch:
            choice = [0]  # Always use first clip in test mode
        else:
            choice = self.rng.choice(
                len(self.clip_ids),
                self.batch_size,
                replace=(self.batch_size > len(self.clip_ids))
            )
        
        # Load or retrieve clips
        if hasattr(self, 'states'):  # Preloaded
            states_batch = self.states[choice]
            actions_batch = self.actions[choice]
        else:  # Load on-the-fly
            states_batch = np.stack([
                load_clip_data(self.root_dir, self.clip_ids[i], "obs")
                for i in choice
            ])
            
            # Determine action key based on mmode
            if self.mmode in ("state_action", "state_intention_controller", "syllable_state_intention"):
                action_key = "ctrl"
            elif self.mmode == "state_intention":
                action_key = "latent_mean"
            elif self.mmode in ("pure_state", "pure_intention"):
                action_key = "obs" if self.mmode == "pure_state" else "latent_mean"
            
            actions_batch = np.stack([
                load_clip_data(self.root_dir, self.clip_ids[i], action_key)
                for i in choice
            ])
        
        # Extract segments
        T = states_batch.shape[1]
        if self.test_batch:
            start = 0
        else:
            max_start = max(0, T - self.seg_len - 1)
            start = self.rng.integers(0, max_start + 1) if max_start > 0 else 0
        
        end = start + self.seg_len + 1
        states_seg = states_batch[:, start:end]  # (B, T+1, D)
        actions_seg = actions_batch[:, start:end]
        
        # Create input/target pairs (autoregressive)
        states_prev = states_seg[:, :-1]  # s_0, ..., s_{T-1}
        states_next = states_seg[:, 1:]   # s_1, ..., s_T
        actions_prev = actions_seg[:, :-1]
        actions_next = actions_seg[:, 1:]
        
        # Extract qpos (local) from full observation
        # obs structure: [ref_obs (470), qpos_local (67), ...rest]
        states_qpos = states_prev[..., 470:537]  # (B, T, 67)
        
        # Add noise augmentation (training only)
        if self.split == "train" and self.noise_std > 0 and self.rng.random() < 0.5:
            states_qpos = states_qpos + self.rng.normal(
                0, self.noise_std, states_qpos.shape
            ).astype(np.float32)
        
        # Prepare inputs/targets based on mmode
        if self.mmode in ("pure_state", "pure_intention"):
            # State/intention prediction
            x_in = states_qpos if self.mmode == "pure_state" else states_prev
            x_tgt = states_next[..., 470:537] if self.mmode == "pure_state" else states_next
        
        elif self.mmode in ("state_action", "state_intention"):
            # Action/intention prediction from state
            x_in = states_qpos
            x_tgt = actions_next
        
        elif self.mmode == "state_intention_controller":
            # Hierarchical: state → intention → action
            x_in = (states_qpos, states_prev)  # (qpos_local, full_obs)
            x_tgt = actions_next
        
        else:
            raise ValueError(f"Unsupported mmode: {self.mmode}")
        
        # Convert to JAX arrays
        return jax.tree.map(jp.asarray, (x_in, x_tgt))


# ✅ Factory function matching your dataloader registry pattern
def make_dataset_h5(
    root_dir: str,
    batch_size: int,
    split: str = "train",
    **kwargs
) -> Iterator[Tuple[jp.ndarray, jp.ndarray]]:
    """
    Create infinite batch generator for HDF5 robotics data.
    
    This matches your existing Datasets["track-mjx"] = make_dataset_h5 pattern.
    
    Usage:
        >>> from ncssm.dataloaders import Datasets
        >>> Datasets["track-mjx"] = make_dataset_h5
        >>> 
        >>> train_loader = Datasets["track-mjx"](
        ...     root_dir="data/clips",
        ...     batch_size=32,
        ...     split="train",
        ...     mmode="state_action",
        ...     seg_len=490
        ... )
        >>> 
        >>> for states, actions in train_loader:
        ...     # Train model
        ...     break
    """
    dataset = RoboticsDataset(
        root_dir=root_dir,
        batch_size=batch_size,
        split=split,
        **kwargs
    )
    return dataset


### Modeling Modes (mmode)

Your project supports multiple modeling paradigms controlled by the `mmode` parameter:

```python
# State → Action (Direct Policy)
mmode = "state_action"
x_in:  (B, T, 67)   # qpos_local states
x_tgt: (B, T, 15)   # actions

# State → State (World Model)
mmode = "pure_state"
x_in:  (B, T, 67)   # qpos_local states
x_tgt: (B, T, 67)   # next states

# State → Intention (Latent Dynamics)
mmode = "state_intention"
x_in:  (B, T, 67)   # qpos_local states
x_tgt: (B, T, 16)   # intention vectors (from VAE encoder)

# Intention → Intention (Pure Latent)
mmode = "pure_intention"
x_in:  (B, T, 16)   # intention vectors
x_tgt: (B, T, 16)   # next intentions

# State → Intention → Action (Hierarchical)
mmode = "state_intention_controller"
x_in:  (qpos_local, full_obs)  # Tuple: (B,T,67), (B,T,537)
x_tgt: (B, T, 15)              # actions

# Syllable + State → Action (Discrete Conditioned)
mmode = "syllable_state_intention"
x_in:  (syllable_onehot, qpos)  # Tuple: (B,T,K), (B,T,67)
x_tgt: (B, T, 15)               # actions
```

**Critical Data Slicing Convention:**
```python
# Full observation structure (537 dims)
obs_full = [
    ref_obs (470),      # Reference trajectory observation
    qpos_local (67),    # Local joint positions (THIS IS YOUR STATE)
    qvel (67),          # Joint velocities
    ...                 # Other proprioceptive data
]

# For modeling, extract qpos_local
state = obs_full[..., 470:537]  # Always slice (B, T, 537) → (B, T, 67)
```

**Why This Matters:**
- Models operate on 67-dim local state, not full 537-dim observation
- Reference trajectory (first 470 dims) handled by encoder separately
- Consistent slicing prevents dimension mismatches


### One-Hot Encoding for Syllables

```python
def _one_hot_safe(ids: np.ndarray, K: int) -> np.ndarray:
    """
    Safe one-hot encoding for syllable IDs.
    
    Handles:
    - Negative IDs → clamp to 0
    - Out-of-range IDs → clamp to K-1
    
    Args:
        ids: Integer syllable IDs (any shape)
        K: Number of syllable classes
    
    Returns:
        One-hot encoded array (*ids.shape, K)
    """
    ids = np.asarray(ids, dtype=np.int32)
    
    # Clamp invalid values
    ids = np.clip(ids, 0, K - 1)
    
    # One-hot encode
    eye = np.eye(K, dtype=np.float32)
    flat = eye[ids.reshape(-1)]
    
    return flat.reshape(*ids.shape, K)


# Usage in dataloader
syllables = load_clip_data(root_dir, clip_idx, "syllables")  # (T,)
syllables_1h = _one_hot_safe(syllables, num_syllables=10)    # (T, 10)
```


### Data Validation


# ✅ DO: Add data validation
def validate_dataset(dataset):
    """Validate dataset integrity."""
    batch_states, batch_actions = next(dataset)
    
    # Check shapes
    B, T, state_dim = batch_states.shape
    assert batch_actions.shape[:-1] == (B, T), \
        f"Shape mismatch: states {batch_states.shape}, actions {batch_actions.shape}"
    
    # Check for NaNs
    assert not jp.any(jp.isnan(batch_states)), "NaN in states"
    assert not jp.any(jp.isnan(batch_actions)), "NaN in actions"
    
    # Check value ranges
    assert jp.all(jp.isfinite(batch_states)), "Inf in states"
    assert jp.all(jp.isfinite(batch_actions)), "Inf in actions"
    
    print(f"✓ Dataset validation passed")
    print(f"  Batch shape: states={batch_states.shape}, actions={batch_actions.shape}")
    print(f"  State range: [{batch_states.min():.3f}, {batch_states.max():.3f}]")
    print(f"  Action range: [{batch_actions.min():.3f}, {batch_actions.max():.3f}]")
```

### Normalization

```python
from typing import Literal, Dict
import jax.numpy as jp
from dataclasses import dataclass

@dataclass
class NormalizationStats:
    """Store normalization statistics."""
    method: Literal["zscore", "minmax"]
    params: Dict[str, jp.ndarray]  # mean/std or min/max/range

class Normalizer:
    """
    Unified normalization interface supporting multiple methods.
    
    CRITICAL: Always save and restore normalization stats with checkpoints.
    """
    
    def __init__(self, method: Literal["zscore", "minmax"] = "zscore"):
        self.method = method
    
    def compute_stats(
        self,
        data: jp.ndarray,
        axis: int | tuple[int, ...] = (0, 1)
    ) -> Dict[str, jp.ndarray]:
        """
        Compute normalization statistics from data.
        
        Args:
            data: Input data of shape (B, T, D) or (B, D)
            axis: Axis/axes to compute statistics over
        
        Returns:
            Dictionary with normalization parameters
        """
        if self.method == "zscore":
            mean = jp.mean(data, axis=axis, keepdims=False)
            std = jp.std(data, axis=axis, keepdims=False)
            std = jp.where(std < 1e-6, jp.ones_like(std), std)  # Prevent div by zero
            return {"mean": mean, "std": std}
        
        elif self.method == "minmax":
            min_val = jp.min(data, axis=axis, keepdims=False)
            max_val = jp.max(data, axis=axis, keepdims=False)
            range_val = max_val - min_val
            range_val = jp.where(range_val < 1e-6, jp.ones_like(range_val), range_val)
            return {"min": min_val, "max": max_val, "range": range_val}
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def normalize(
        self,
        data: jp.ndarray,
        stats: Dict[str, jp.ndarray]
    ) -> jp.ndarray:
        """Apply normalization."""
        if self.method == "zscore":
            return (data - stats["mean"]) / stats["std"]
        elif self.method == "minmax":
            return (data - stats["min"]) / stats["range"]
    
    def denormalize(
        self,
        data: jp.ndarray,
        stats: Dict[str, jp.ndarray]
    ) -> jp.ndarray:
        """Reverse normalization."""
        if self.method == "zscore":
            return data * stats["std"] + stats["mean"]
        elif self.method == "minmax":
            return data * stats["range"] + stats["min"]


# ✅ DO: Compute stats on representative sample
def compute_normalization_stats(
    dataset,
    n_batches: int = 100,
    normalizer: Normalizer = None
) -> tuple[Dict, Dict]:
    """
    Compute normalization statistics from dataset.
    
    CRITICAL: Do this BEFORE adding noise augmentation to training data.
    """
    if normalizer is None:
        normalizer = Normalizer(method="zscore")
    
    state_samples = []
    action_samples = []
    
    print(f"Computing normalization from {n_batches} batches...")
    
    for i in range(n_batches):
        states, actions = next(dataset)
        state_samples.append(states)
        action_samples.append(actions)
    
    all_states = jp.concatenate(state_samples, axis=0)
    all_actions = jp.concatenate(action_samples, axis=0)
    
    state_stats = normalizer.compute_stats(all_states)
    action_stats = normalizer.compute_stats(all_actions)
    
    # Print statistics
    if normalizer.method == "zscore":
        print(f"State: μ ∈ [{state_stats['mean'].min():.3f}, {state_stats['mean'].max():.3f}], "
              f"σ ∈ [{state_stats['std'].min():.3f}, {state_stats['std'].max():.3f}]")
        print(f"Action: μ ∈ [{action_stats['mean'].min():.3f}, {action_stats['mean'].max():.3f}], "
              f"σ ∈ [{action_stats['std'].min():.3f}, {action_stats['std'].max():.3f}]")
    else:
        print(f"State: range [{state_stats['min'].min():.3f}, {state_stats['max'].max():.3f}]")
        print(f"Action: range [{action_stats['min'].min():.3f}, {action_stats['max'].max():.3f}]")
    
    return state_stats, action_stats
```

## 🧠 Model Architecture Standards

### State Space Models (S5/Mamba)

```python
from flax import linen as nn
import jax.numpy as jp
from typing import Callable, Optional

class SequenceModel(nn.Module):
    """
    Base class for sequence models.
    
    Conventions:
    - Input: (B, T, D_in) where B=batch, T=time, D=features
    - Output: (B, T, D_out)
    - Always handle variable sequence lengths
    """
    
    n_layers: int
    d_model: int
    d_output: int
    dropout: float = 0.0
    training: bool = False
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = False) -> jp.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input sequence (B, T, D_in)
            train: Whether in training mode (for dropout)
        
        Returns:
            Output sequence (B, T, D_out)
        """
        raise NotImplementedError


class S5Model(SequenceModel):
    """
    S5 sequence model with proper initialization.
    
    Key design decisions:
    - Use diagonal state space structure for efficiency
    - Complex-valued parameters for modeling oscillations
    - Proper discretization method (ZOH, bilinear, etc.)
    """
    
    ssm_init: Callable
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = False) -> jp.ndarray:
        B, T, D_in = x.shape
        
        # Optional input projection
        if D_in != self.d_model:
            x = nn.Dense(self.d_model, name="input_proj")(x)
        
        # Stack S5 layers
        for i in range(self.n_layers):
            # S5 layer (state space block)
            x_ssm = self.ssm_init(
                name=f"s5_layer_{i}"
            )(x)
            
            # Residual connection
            x = x + x_ssm
            
            # Optional dropout
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        
        # Output projection
        x = nn.Dense(self.d_output, name="output_proj")(x)
        
        return x


# ✅ DO: Use proper model initialization
def create_s5_model(
    state_dim: int,
    action_dim: int,
    n_layers: int = 4,
    d_model: int = 256,
    ssm_size: int = 64,
    **ssm_kwargs
) -> S5Model:
    """
    Create S5 model with proper initialization.
    
    Args:
        state_dim: Input state dimension
        action_dim: Output action dimension
        n_layers: Number of S5 layers
        d_model: Hidden dimension
        ssm_size: State space dimension (N)
        **ssm_kwargs: Additional SSM initialization args
    
    Returns:
        Initialized S5 model
    """
    from functools import partial
    
    # S5 SSM initialization with sensible defaults
    ssm_init = partial(
        init_S5SSM,
        ssm_size=ssm_size,
        d_model=d_model,
        C_init="lecun_normal",
        discretization="zoh",  # Zero-order hold
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=True,  # Use conjugate symmetry
        clip_eigs=False,
        bidirectional=False,
        **ssm_kwargs
    )
    
    model = S5Model(
        n_layers=n_layers,
        d_model=d_model,
        d_output=action_dim,
        ssm_init=ssm_init,
        dropout=0.1,
        training=True
    )
    
    return model
```

### Model Registry Pattern

```python
from typing import Dict, Type, Any, Callable
from functools import partial

# Global model registry
MODEL_REGISTRY: Dict[str, Type] = {}

def register_model(name: str) -> Callable:
    """Decorator to register models."""
    def decorator(cls: Type) -> Type:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


@register_model("s5")
class S5ARModel(nn.Module):
    """S5 autoregressive model."""
    # Implementation...
    pass


@register_model("mamba")
class MambaARModel(nn.Module):
    """Mamba autoregressive model."""
    # Implementation...
    pass


def create_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Factory function to create models from config.
    
    Args:
        model_type: Model name (must be in registry)
        config: Model configuration
    
    Returns:
        Initialized model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_cls = MODEL_REGISTRY[model_type]
    return model_cls(**config)


# ✅ Usage in training script
def main(cfg: Dict[str, Any]):
    model = create_model(
        model_type=cfg["model"]["type"],
        config=cfg["model"]["params"]
    )
```

## 🔧 Configuration Management

### YAML Configuration Structure

Your project uses YAML files in `configs/` for all experiments:

```yaml
# configs/s5_sa.yaml (State → Action baseline)

# Model configuration
model_type: "s5"  # or "mamba"
mmode: "state_action"

# Architecture
n_layers: 6
d_model: 256
ssm_size: 64  # For S5

# S5-specific initialization
c_init: "lecun_normal"
discretization: "zoh"
dt_min: 0.001
dt_max: 0.1
conj_sym: true
clip_eigs: false
bidirectional: false
bandlimit: 0.0

# Training hyperparameters
epochs: 200
batch_size: 32
seg_len: 490
learning_rate: 0.001
grad_clip: 1.0
dropout: 0.1
noise_std: 0.01  # Input augmentation

# Data paths
dataset: "track-mjx"
root_dir: "data/clips"
train_frac: 0.9
preload: false
clip_idx: 25  # For test_batch mode

# Normalization
norm_method: "minmax"  # or "zscore"

# Dimensions (must match data)
state_dim: 67   # qpos_local
action_dim: 15  # control
int_dim: 16     # latent (VAE)

# Checkpointing
ckpt_dir: "checkpoints/s5_sa"
resume_ckpt_dir: null
keep_ckpts: 3

# Logging (W&B)
project: "ncssm"
name: "s5-state-action"
tags: ["s5", "baseline"]
grad_freq: 10
action_video_freq: 10

# Testing/debugging
test_batch: false
test_full_seq: false
iters_per_epoch: 1000
seed: 0
```

### Config Loading Pattern

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate YAML configuration."""
    config_path = Path(config_path).expanduser().resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ["model_type", "mmode", "n_layers", "state_dim", "action_dim"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    
    # Set defaults
    config.setdefault("seed", 0)
    config.setdefault("dropout", 0.0)
    config.setdefault("grad_clip", 1.0)
    config.setdefault("norm_method", "minmax")
    
    print(f"✓ Loaded config from {config_path}")
    print(f"  Model: {config['model_type']}, Mode: {config['mmode']}")
    print(f"  State dim: {config['state_dim']}, Action dim: {config['action_dim']}")
    
    return config


# ✅ CLI with overrides
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*",
                        help="Override: key=value key2=value2")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Apply CLI overrides
    if args.override:
        for item in args.override:
            key, val = item.split("=")
            try:
                cfg[key] = eval(val)  # Try Python literal
            except:
                cfg[key] = val  # Fallback to string
            print(f"  Override: {key} = {cfg[key]}")
    
    train_model(cfg)

# Usage:
# python train.py --config configs/s5_sa.yaml
# python train.py --config configs/s5_sa.yaml --override "learning_rate=0.0005" "batch_size=64"
```

## 🎓 Training Loop Standards

### Training State

```python
from flax.training import train_state
import optax

class TrainStateWithStats(train_state.TrainState):
    """
    Extended training state with normalization statistics.
    
    CRITICAL: Always include normalization stats in checkpoints
    to ensure inference matches training.
    """
    
    state_stats: Dict[str, jp.ndarray]
    action_stats: Dict[str, jp.ndarray]
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, state_stats, action_stats, **kwargs):
        """Create training state with normalization stats."""
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            state_stats=state_stats,
            action_stats=action_stats,
            **kwargs
        )


# ✅ DO: Create training state with all necessary components
def setup_training(
    model: nn.Module,
    config: Dict[str, Any],
    state_stats: Dict,
    action_stats: Dict
) -> TrainStateWithStats:
    """
    Initialize training state.
    
    Args:
        model: Flax model
        config: Training configuration
        state_stats: State normalization statistics
        action_stats: Action normalization statistics
    
    Returns:
        Training state ready for training
    """
    # Initialize parameters
    rng = jax.random.PRNGKey(config["seed"])
    sample_input = jp.zeros((1, 1, config["state_dim"]))
    params = model.init(rng, sample_input)["params"]
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.get("grad_clip", 1.0)),
        optax.adam(
            learning_rate=config["learning_rate"],
            b1=config.get("adam_b1", 0.9),
            b2=config.get("adam_b2", 0.999)
        )
    )
    
    # Create training state
    state = TrainStateWithStats.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        state_stats=state_stats,
        action_stats=action_stats
    )
    
    return state
```

### Training Loop

```python
import wandb
from tqdm import trange
from pathlib import Path

def train_model(
    state: TrainStateWithStats,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    normalizer: Normalizer
) -> TrainStateWithStats:
    """
    Main training loop.
    
    Design principles:
    - Separate JITed step functions from I/O
    - Log metrics every N steps
    - Validate every epoch
    - Save checkpoints periodically
    - Clear caches to prevent memory issues
    """
    
    # Setup
    ckpt_dir = Path(config["ckpt_dir"]).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    run = wandb.init(
        project=config.get("project", "robotics"),
        name=config.get("name", "experiment"),
        config=config,
        tags=config.get("tags", [])
    )
    
    # JITed training step
    @jax.jit
    def train_step(state, batch_states, batch_actions):
        def loss_fn(params):
            # Normalize inputs
            states_norm = normalizer.normalize(batch_states, state.state_stats)
            actions_norm = normalizer.normalize(batch_actions, state.action_stats)
            
            # Forward pass
            preds = state.apply_fn({"params": params}, states_norm, train=True)
            
            # Compute loss
            loss = jp.mean((preds - actions_norm) ** 2)
            
            return loss, {"mse": loss}
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, metrics, grads
    
    @jax.jit
    def val_step(params, batch_states, batch_actions):
        # Normalize
        states_norm = normalizer.normalize(batch_states, state.state_stats)
        actions_norm = normalizer.normalize(batch_actions, state.action_stats)
        
        # Forward pass (no dropout)
        preds = state.apply_fn({"params": params}, states_norm, train=False)
        
        # Compute loss
        loss = jp.mean((preds - actions_norm) ** 2)
        
        return loss
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, config["epochs"] + 1):
        # Training
        train_losses = []
        
        for step in trange(config["steps_per_epoch"], desc=f"Epoch {epoch}", leave=False):
            batch_states, batch_actions = next(train_loader)
            state, metrics, grads = train_step(state, batch_states, batch_actions)
            
            train_losses.append(float(metrics["mse"]))
            
            # Log gradients periodically
            if step % config.get("grad_log_freq", 100) == 0:
                grad_norm = optax.global_norm(grads)
                wandb.log({"train/grad_norm": float(grad_norm)}, step=state.step)
        
        train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        val_losses = []
        for _ in range(config.get("val_steps", 10)):
            batch_states, batch_actions = next(val_loader)
            val_loss = val_step(state.params, batch_states, batch_actions)
            val_losses.append(float(val_loss))
        
        val_loss = sum(val_losses) / len(val_losses)
        
        # Logging
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/lr": float(config["learning_rate"])
        }, step=epoch)
        
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4e} | Val: {val_loss:.4e}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_dir / "best", state, epoch)
        
        if epoch % config.get("ckpt_freq", 10) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch}", state, epoch)
        
        # Clear JAX cache periodically
        if epoch % 50 == 0:
            jax.clear_caches()
    
    wandb.finish()
    return state


def save_checkpoint(path: Path, state: TrainStateWithStats, epoch: int):
    """Save checkpoint with all necessary state."""
    from flax.training import checkpoints
    
    path.mkdir(parents=True, exist_ok=True)
    checkpoints.save_checkpoint(
        str(path),
        state,
        epoch,
        keep=3,  # Keep last 3 checkpoints
        orbax_checkpointer=None
    )
    print(f"✓ Saved checkpoint to {path}")
```

## 🔍 Inference Standards

### Predictor Interface

```python
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    """
    Abstract base class for all predictors.
    
    Design principles:
    - Load checkpoint in __init__
    - Expose simple predict() interface
    - Handle normalization internally
    - Support batched and single predictions
    """
    
    def __init__(self, config: Dict[str, Any], global_config: Optional[Dict] = None):
        self.config = config
        self.global_config = global_config
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Load model, checkpoint, and setup prediction function."""
        pass
    
    @abstractmethod
    def predict(self, obs: jp.ndarray, **kwargs) -> jp.ndarray:
        """Predict actions from observations."""
        pass


class S5Predictor(BasePredictor):
    """S5 model predictor for inference."""
    
    def _setup(self) -> None:
        """Load S5 checkpoint and create prediction function."""
        
        # Load checkpoint
        ckpt_dir = Path(self.config["ckpt_dir"]).expanduser().resolve()
        state = load_checkpoint(ckpt_dir)
        
        self.params = state.params
        self.apply_fn = state.apply_fn
        self.state_stats = state.state_stats
        self.action_stats = state.action_stats
        
        # Create normalizer
        norm_method = "zscore" if "mean" in self.state_stats else "minmax"
        self.normalizer = Normalizer(norm_method)
        
        # Context window for autoregressive prediction
        self.context_window = self.config.get("context_window", 0)
        self.context_buffer = None
        
        # JIT prediction function
        @jax.jit
        def _predict_fn(params, obs_norm):
            """JITed prediction."""
            preds = self.apply_fn(
                {"params": params},
                obs_norm[None, :, :],  # Add batch dim
                train=False
            )
            return preds.squeeze(0)  # Remove batch dim
        
        self._predict_fn = _predict_fn
    
    def predict(
        self,
        obs: jp.ndarray,
        update_context: bool = True
    ) -> jp.ndarray:
        """
        Predict action from observation.
        
        Args:
            obs: Observation (D,) or (B, D)
            update_context: Whether to update context buffer
        
        Returns:
            Predicted action (A,) or (B, A)
        """
        # Ensure batch dimension
        single_input = obs.ndim == 1
        if single_input:
            obs = obs[None, :]  # (D,) -> (1, D)
        
        # Normalize observation
        obs_norm = self.normalizer.normalize(obs, self.state_stats)
        
        # Add to context window if using autoregressive prediction
        if self.context_window > 0:
            if self.context_buffer is None:
                # Initialize context buffer
                self.context_buffer = jp.zeros((self.context_window, obs_norm.shape[-1]))
            
            # Update context: shift and append
            self.context_buffer = jp.concatenate([
                self.context_buffer[1:],  # Remove oldest
                obs_norm  # Add newest
            ], axis=0)
            
            input_seq = self.context_buffer
        else:
            input_seq = obs_norm
        
        # Predict
        preds_norm = self._predict_fn(self.params, input_seq)
        
        # Denormalize
        preds = self.normalizer.denormalize(preds_norm, self.action_stats)
        
        # Remove batch dim if input was single
        if single_input:
            preds = preds.squeeze(0)
        
        return preds
    
    def reset(self):
        """Reset context buffer."""
        self.context_buffer = None


def load_checkpoint(ckpt_dir: Path) -> TrainStateWithStats:
    """
    Load checkpoint for inference.
    
    CRITICAL: Must restore exact same TrainStateWithStats structure
    including normalization stats.
    """
    from flax.training import checkpoints
    
    # Find latest checkpoint
    latest = checkpoints.latest_checkpoint(str(ckpt_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    
    print(f"Loading checkpoint: {latest}")
    
    # Create dummy state for restoration
    # (actual values will be overwritten)
    dummy_state = create_dummy_train_state()
    
    # Restore
    state = checkpoints.restore_checkpoint(str(ckpt_dir), target=dummy_state)
    
    return state
```

## 📈 Experiment Tracking

### W&B Best Practices

```python
import wandb
import matplotlib.pyplot as plt
import numpy as np

# ✅ DO: Initialize W&B with comprehensive config
def init_wandb(config: Dict[str, Any]) -> wandb.Run:
    """Initialize Weights & Biases tracking."""
    
    # Add system info
    config["system"] = {
        "hostname": os.uname().nodename,
        "jax_version": jax.__version__,
        "jax_devices": str(jax.devices()),
        "python_version": sys.version
    }
    
    run = wandb.init(
        project=config["project"],
        entity=config.get("entity"),
        name=config.get("name", f"run_{datetime.now():%Y%m%d_%H%M%S}"),
        config=config,
        tags=config.get("tags", []),
        notes=config.get("notes"),
        dir=config.get("wandb_dir", "./wandb")
    )
    
    return run


# ✅ DO: Log rich visualizations
def log_predictions(
    state: TrainStateWithStats,
    dataset,
    epoch: int,
    n_samples: int = 5,
    max_timesteps: int = 150
):
    """Log prediction visualizations to W&B."""
    
    # Get sample
    states, actions_true = next(dataset)
    
    # Predict
    states_norm = state.normalizer.normalize(states, state.state_stats)
    actions_pred_norm = state.apply_fn(
        {"params": state.params},
        states_norm,
        train=False
    )
    actions_pred = state.normalizer.denormalize(
        actions_pred_norm,
        state.action_stats
    )
    
    # Convert to numpy
    states_np = np.array(states[0, :max_timesteps, :])
    actions_true_np = np.array(actions_true[0, :max_timesteps, :])
    actions_pred_np = np.array(actions_pred[0, :max_timesteps, :])
    
    # Plot first N action dimensions
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2*n_samples))
    timesteps = np.arange(len(actions_true_np))
    
    for i in range(n_samples):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(timesteps, actions_true_np[:, i], label="Ground Truth", linewidth=2)
        ax.plot(timesteps, actions_pred_np[:, i], '--', label="Predicted", linewidth=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"Action Dim {i}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Log to W&B
    wandb.log({
        "predictions/actions": wandb.Image(fig),
        "epoch": epoch
    })
    plt.close()


# ✅ DO: Log gradient statistics
def log_gradient_stats(grads, epoch: int):
    """Log gradient statistics to W&B."""
    
    # Flatten all gradients
    flat_grads = jax.tree_util.tree_leaves(grads)
    all_grads = jp.concatenate([g.flatten() for g in flat_grads])
    
    grad_stats = {
        "grads/mean": float(jp.mean(jp.abs(all_grads))),
        "grads/std": float(jp.std(all_grads)),
        "grads/max": float(jp.max(jp.abs(all_grads))),
        "grads/norm": float(jp.linalg.norm(all_grads)),
        "epoch": epoch
    }
    
    wandb.log(grad_stats)
    
    # Check for gradient issues
    if jp.any(jp.isnan(all_grads)):
        wandb.alert(
            title="NaN in Gradients",
            text=f"NaN detected in gradients at epoch {epoch}",
            level=wandb.AlertLevel.ERROR
        )
```

## 🧪 Testing ML Code

### Unit Tests for Models

```python
import pytest
import jax
import jax.numpy as jp

def test_model_shapes():
    """Test that model produces correct output shapes."""
    
    # Setup
    model = S5ARModel(
        n_layers=2,
        action_dim=15,
        state_dim=67,
        ssm_ctor=lambda: init_S5SSM(...),
        dropout=0.0,
        training=False
    )
    
    rng = jax.random.PRNGKey(0)
    batch_size, seq_len, state_dim = 4, 100, 67
    x = jp.zeros((batch_size, seq_len, state_dim))
    
    # Initialize
    params = model.init(rng, x)["params"]
    
    # Forward pass
    output = model.apply({"params": params}, x, train=False)
    
    # Assertions
    assert output.shape == (batch_size, seq_len, 15), \
        f"Expected (4, 100, 15), got {output.shape}"
    assert not jp.any(jp.isnan(output)), "NaN in output"
    assert jp.all(jp.isfinite(output)), "Inf in output"


def test_normalization_invertibility():
    """Test that normalization is invertible."""
    
    # Create sample data
    data = jax.random.normal(jax.random.PRNGKey(0), (100, 50, 10))
    
    # Compute stats and normalize
    normalizer = Normalizer("zscore")
    stats = normalizer.compute_stats(data)
    normalized = normalizer.normalize(data, stats)
    reconstructed = normalizer.denormalize(normalized, stats)
    
    # Check reconstruction
    assert jp.allclose(data, reconstructed, atol=1e-5), \
        "Normalization not invertible"


def test_train_step_decreases_loss():
    """Test that training step reduces loss."""
    
    # Setup
    state = setup_training(model, config, state_stats, action_stats)
    batch_states, batch_actions = get_dummy_batch()
    
    # Initial loss
    initial_loss = compute_loss(state.params, batch_states, batch_actions)
    
    # Training step
    state, metrics, _ = train_step(state, batch_states, batch_actions)
    
    # New loss
    new_loss = compute_loss(state.params, batch_states, batch_actions)
    
    # Loss should decrease (or at least not increase significantly)
    assert new_loss <= initial_loss * 1.1, \
        f"Loss increased: {initial_loss:.4e} -> {new_loss:.4e}"
```

## 🎯 Code Style for ML

### Naming Conventions

```python
# States and observations
obs, state, x           # Current observation/state
next_obs, next_state    # Next timestep
obs_history             # Historical observations

# Actions
action, a, u            # Control action
pred_action             # Predicted action

# Dimensions
B, T, D                 # Batch, Time, Feature dimension
state_dim, action_dim   # Input/output dimensions
d_model, d_hidden       # Model hidden dimensions

# Parameters
params                  # Model parameters
state                   # Training state (TrainState)
config, cfg             # Configuration dictionary

# Normalization
*_norm                  # Normalized version (e.g., obs_norm)
*_stats                 # Normalization statistics (e.g., state_stats)

# Losses and metrics
loss, mse, mae          # Loss values
grads, gradients        # Gradient values
```

### Documentation Style

```python
def train_sequence_model(
    model: nn.Module,
    dataset: Iterator,
    config: Dict[str, Any],
    normalizer: Normalizer
) -> TrainStateWithStats:
    """
    Train a sequence model with teacher forcing.
    
    Architecture assumptions:
    - Model takes (B, T, D_in) and outputs (B, T, D_out)
    - Supports variable sequence lengths
    - Uses dropout during training
    
    Args:
        model: Flax model (S5, Mamba, etc.)
        dataset: Iterator yielding (states, actions) batches
        config: Training configuration with keys:
            - epochs: Number of training epochs
            - steps_per_epoch: Training steps per epoch
            - learning_rate: Adam learning rate
            - grad_clip: Gradient clipping threshold
        normalizer: Normalizer for states and actions
    
    Returns:
        Trained model state with parameters and stats
    
    Raises:
        ValueError: If config missing required keys
        RuntimeError: If training diverges (NaN in loss)
    
    Example:
        >>> config = {
        ...     "epochs": 100,
        ...     "steps_per_epoch": 1000,
        ...     "learning_rate": 1e-3
        ... }
        >>> state = train_sequence_model(model, dataset, config, normalizer)
    """
```

## ⚠️ Common Pitfalls

### JAX-Specific Issues

```python
# ❌ DON'T: Mutate arrays (JAX arrays are immutable)
def bad_update(x):
    x[0] = 10  # Error! JAX arrays are immutable
    return x

# ✅ DO: Use .at[] for updates
def good_update(x):
    return x.at[0].set(10)


# ❌ DON'T: Use Python control flow inside JIT
@jax.jit
def bad_conditional(x, threshold):
    if x > threshold:  # ❌ Python if not traceable
        return x * 2
    return x

# ✅ DO: Use JAX control flow
@jax.jit
def good_conditional(x, threshold):
    return jax.lax.cond(
        x > threshold,
        lambda x: x * 2,
        lambda x: x,
        x
    )


# ❌ DON'T: Forget to handle NaN/Inf
def bad_loss(pred, target):
    return jp.mean((pred - target) ** 2)

# ✅ DO: Add guards against NaN/Inf
def good_loss(pred, target):
    mse = jp.mean((pred - target) ** 2)
    
    # Guard against NaN/Inf
    is_valid = jp.isfinite(mse)
    return jax.lax.cond(
        is_valid,
        lambda: mse,
        lambda: jp.array(1e6)  # Large penalty for invalid loss
    )
```

### Normalization Issues

```python
# ❌ DON'T: Compute stats on augmented training data
train_loader = create_loader(noise_std=0.1)  # With augmentation
stats = compute_stats(train_loader)  # ❌ Stats are biased

# ✅ DO: Compute stats on clean data
clean_loader = create_loader(noise_std=0.0)  # No augmentation
stats = compute_stats(clean_loader)  # ✅ Correct
train_loader = create_loader(noise_std=0.1)  # Now add augmentation


# ❌ DON'T: Forget to save normalization stats
checkpoints.save_checkpoint(ckpt_dir, state.params, epoch)  # ❌ Lost stats

# ✅ DO: Save complete training state
checkpoints.save_checkpoint(ckpt_dir, state, epoch)  # ✅ Includes stats
```

### Memory Issues

```python
# ❌ DON'T: Accumulate gradients in Python list
grads_list = []
for batch in dataloader:
    _, grads = train_step(state, batch)
    grads_list.append(grads)  # ❌ Keeps all on device

# ✅ DO: Convert to numpy or aggregate immediately
grad_norms = []
for batch in dataloader:
    _, grads = train_step(state, batch)
    grad_norm = float(optax.global_norm(grads))  # Convert to Python float
    grad_norms.append(grad_norm)  # ✅ Only scalar on host
```

## 🐛 Project-Specific Gotchas

### Issue 1: Observation Slicing

```python
# ❌ WRONG: Using full 537-dim observation
obs_full = next(dataloader)  # (B, T, 537)
model_input = obs_full  # ❌ Wrong dimension!

# ✅ CORRECT: Extract qpos_local (67-dim)
obs_full = next(dataloader)  # (B, T, 537)
qpos_local = obs_full[..., 470:537]  # ✅ (B, T, 67)
model_input = qpos_local
```

**Why this matters:** Your models are trained on 67-dim qpos, not full obs. If you pass wrong dimensions at inference, predictions will be garbage.

### Issue 2: Normalization Stats Must Be Saved

```python
# ❌ BAD: Only save model parameters
checkpoints.save_checkpoint(ckpt_dir, state.params, epoch)

# ✅ GOOD: Save complete TrainStateWithStats
checkpoints.save_checkpoint(ckpt_dir, state, epoch)  # Includes state_stats, action_stats
```

**Why this matters:** Without normalization stats, inference will use wrong scaling and produce incorrect predictions.

### Issue 3: Compute Norm Stats on Clean Data

```python
# ❌ WRONG: Compute stats after adding noise
train_loader = make_dataset_h5(..., noise_std=0.01)  # With noise
state_stats = compute_stats(train_loader)  # ❌ Biased by noise!

# ✅ CORRECT: Compute stats on clean data first
clean_loader = make_dataset_h5(..., noise_std=0.0)  # No noise
state_stats = compute_stats(clean_loader)  # ✅ Correct

# Then create noisy loader for training
train_loader = make_dataset_h5(..., noise_std=0.01)  # Now add noise
```

**Why this matters:** Your normalization should capture the true data distribution, not the augmented one.

### Issue 4: Context Window Management

```python
# For autoregressive prediction with context window
class S5Predictor:
    def __init__(self, config):
        self.context_window = config.get("context_dim", 50)
        self.context_buffer = None  # Initialize as None
    
    def predict(self, obs, update_context=True):
        if self.context_window > 0:
            if self.context_buffer is None:
                # First call: initialize buffer
                self.context_buffer = jp.zeros((self.context_window, obs.shape[-1]))
            
            # Update: shift old, add new
            self.context_buffer = jp.concatenate([
                self.context_buffer[1:],  # Drop oldest
                obs[None, :]  # Add newest
            ], axis=0)
            
            input_seq = self.context_buffer  # (context_window, D)
        else:
            input_seq = obs[None, :]  # (1, D)
        
        # Predict on full sequence
        pred = self.model(input_seq[None, :, :])  # (1, T, D)
        return pred[0, -1, :]  # Return only last timestep
    
    def reset(self):
        """Call this between rollouts!"""
        self.context_buffer = None
```

### Issue 5: MMode Confusion

```python
# Your different mmodes require different handling

def create_predictor(config):
    mmode = config["mmode"]
    
    if mmode in ("state_action", "state_intention"):
        # Simple: state → action/intention
        return S5Predictor(config)
    
    elif mmode == "state_intention_controller":
        # Hierarchical: state → intention → action (needs decoder)
        return S5Controller(config)  # Different class!
    
    elif mmode == "syllable_state_intention":
        # Requires syllable predictor too
        return S5SyllableController(config)  # Different class!
    
    elif mmode in ("pure_state", "pure_intention"):
        # World model: state → next_state
        return S5WorldModel(config)  # Different class!
    
    else:
        raise ValueError(f"Unknown mmode: {mmode}")
```

### Issue 6: Test Batch vs Regular Training

```python
# test_batch=True uses SAME clip every iteration (for overfitting test)
# Make sure you understand which mode you're in!

if config["test_batch"]:
    print("⚠️ WARNING: test_batch=True, using single clip for overfitting test")
    print(f"  Using clip_idx={config['clip_idx']}")
    # Expect training loss to go to near zero
    # If it doesn't, something is broken

else:
    print("Regular training mode with random sampling")
    # Training loss won't go to zero (generalization)
```

### Issue 7: Seg_len vs Full Sequence

```python
# Your data is loaded in segments (default seg_len=490)
# But full clips can be ~500-3000 frames

# Training: Use seg_len for manageable sequences
train_loader = make_dataset_h5(..., seg_len=490)

# Evaluation: Consider using longer sequences or full clips
val_loader = make_dataset_h5(..., seg_len=1000)  # Longer

# Full sequence test (if test_full_seq=True)
# Model gets entire clip at once (B, full_T, D)
# Only works if clip fits in memory
```

## 📚 Additional Resources

### JAX Resources
- JAX Documentation: https://jax.readthedocs.io/
- JAX GitHub: https://github.com/google/jax
- Flax Documentation: https://flax.readthedocs.io/

### State Space Models
- S5 Paper: https://arxiv.org/abs/2208.04933
- Mamba Paper: https://arxiv.org/abs/2312.00752
- Structured State Spaces: https://github.com/state-spaces/s4

### MuJoCo and Robotics
- MuJoCo Documentation: https://mujoco.readthedocs.io/
- Brax (JAX physics): https://github.com/google/brax
- DMC (DeepMind Control): https://github.com/deepmind/dm_control

---

## 🔄 Version Control for ML

### Git Workflow for Experiments

```bash
# Create experiment branch
git checkout -b exp/s5-baseline
git checkout -b exp/mamba-comparison
git checkout -b exp/ablation-ssm-size

# Commit with descriptive messages
git commit -m "feat(s5): add S5 baseline model with 4 layers"
git commit -m "exp: test SSM size ablation (64, 128, 256)"
git commit -m "fix: correct normalization stats computation"

# Tag successful experiments
git tag -a exp-s5-best-val -m "Best S5 model: val_loss=0.0042"
```

### .gitignore for ML Projects

```gitignore
# Checkpoints and weights
*.ckpt
*.pkl
*.pth
checkpoints/
wandb/

# Data
data/
datasets/
*.h5
*.hdf5

# Outputs
outputs/
results/
videos/

# Python
__pycache__/
*.pyc
.pytest_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
```

---

**Remember**: ML code should be as reproducible, testable, and maintainable as any other software. Good engineering practices apply to research code too!

---

## 🎯 Quick Reference Cheat Sheet

### Data Dimensions

```python
# Observation structure
obs_full: (B, T, 537)
├── ref_obs: [0:470]      # Reference trajectory (470)
├── qpos_local: [470:537] # YOUR STATE (67) ← Use this!
└── others: [537:]        # Velocities, etc.

# Model I/O
state_action:      (B, T, 67) → (B, T, 15)
state_intention:   (B, T, 67) → (B, T, 16)
pure_state:        (B, T, 67) → (B, T, 67)
```

### Critical Reminders

1. **Always slice observations**: `obs[..., 470:537]` for 67-dim state
2. **Compute norm stats on clean data** (noise_std=0.0)
3. **Save TrainStateWithStats** not just params (includes norm stats)
4. **Reset context buffer** between rollouts
5. **Check mmode** - different modes need different predictors
6. **Use jax.debug.print** inside JIT functions
7. **Clear JAX caches** every ~50 epochs for long training

### Common Commands

```bash
# Training
python train.py --config configs/s5_sa.yaml
python trainer/deep_ssm/s5_trainer.py  # Direct

# With overrides
python train.py --config configs/s5_sa.yaml \
    --override "learning_rate=0.0005" "batch_size=64"

# Evaluation
python rollout.py --config configs/rollout.yaml

# Test overfitting (single clip)
python train.py --config configs/s5_sa.yaml \
    --override "test_batch=True" "clip_idx=25"
```

### Debugging Checklist

- [ ] Shape assertions in forward pass
- [ ] NaN checks after each operation
- [ ] Gradient norm logging (grad_freq)
- [ ] Normalization stats printed
- [ ] Checkpoint includes state_stats, action_stats
- [ ] Context buffer reset between episodes
- [ ] Correct mmode for task
- [ ] Observation slicing correct [470:537]

### File Locations

```
checkpoints/{experiment_name}/
├── checkpoint_{epoch}  # Full TrainStateWithStats
└── best/              # Best validation checkpoint

configs/
├── s5_sa.yaml         # State → Action
├── s5_si.yaml         # State → Intention
├── s5_sic.yaml        # State → Int → Action
└── rollout.yaml       # Evaluation config

data/clips/
├── clip_0.h5          # /obs, /ctrl, /latent_mean
├── clip_1.h5
└── ...
```

---

_This guide is specific to the ncssm robotics project. Update as patterns evolve._
