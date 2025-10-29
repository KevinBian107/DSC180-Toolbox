# CLAUDE.md

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
x_tgt: (B, T, 38)   # actions

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
x_tgt: (B, T, 38)              # actions

# Syllable + State → Action (Discrete Conditioned)
mmode = "syllable_state_intention"
x_in:  (syllable_onehot, qpos)  # Tuple: (B,T,K), (B,T,67)
x_tgt: (B, T, 38)               # actions
```

**Critical Data Slicing Convention:**
```python
# Full observation structure (537 dims)
obs_full = [
    ref_obs (470),      # Reference trajectory observation
    qpos_local (67),    # Local joint positions (THIS IS YOUR STATE)
    ...                 # Other proprioceptive data
]

# For modeling, extract qpos_local
state = obs_full[..., 470:537]  # Always slice (B, T, 537) → (B, T, 67)
```

**Why This Matters:**
- Models operate on 67-dim local state, not full 537-dim observation
- Reference trajectory (first 470 dims) handled by encoder separately
- Consistent slicing prevents dimension mismatches


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
        config: Training configuration
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


## 📚 Additional Resources

### JAX Resources
- JAX Documentation: https://jax.readthedocs.io/
- JAX GitHub: https://github.com/google/jax
- Flax Documentation: https://flax.readthedocs.io/
- MuJoCo Documentation: https://mujoco.readthedocs.io/

### State Space Models
- S5 Paper: https://arxiv.org/abs/2208.04933
- Mamba Paper: https://arxiv.org/abs/2312.00752
- Keypoint Moseq GitHub: https://github.com/dattalab/keypoint-moseq
- Jax Moseq GitHub: https://github.com/dattalab/jax-moseq

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