# Build Scripts

[Back to home](/README.md)

***Notes copied from dsc [180A capstone website](https://dsc-capstone.org/2025-26/lessons/03/#build-scripts)***

As alluded to in the previous sections, it’s desirable for our project to be easy to run – ideally, future users can run `run project` in their Terminal and see the output of your work. That’s precisely what build scripts enable! Specifically, a build script runs the code in a project to “build” desired output. A build script does not contain the implementation details of a project – that belongs in library code. Instead, **a build script calls library code, and thus should not be very complicated**.

Build scripts are important for ensuring that our work is reproducible, as all science should be. While they are still not very common in data science projects, build scripts have been around for decades in other contexts (one example of such is `Makefiles`, which are used to coordinate the compilation of C/C++ code and, more generally, to run general Bash (Terminal) commands).

To keep things simple, we should create a **barebones build script** from scratch, called `run.py`. Note that this is really just a simple python file that automates our work flows. If we like, we can write a build script in another language, e.g. `run.R` for R projects, or `run.sh` in Bash.

## Run.py
A target specifies what to build. Specifically, a target is a string describing the desired output, and targets are used when calling build scripts from the Terminal. We should create targets for all major “steps” in our project pipeline, particular for steps that it would make sense to run in isolation of other steps. For instance:

- Target called `data` that prepares the data for the project by downloading data and running our ETL code. To use this target, users would run python `run.py data` in the Terminal.

- Target called `features` that builds the features for the project, from the already-processed data. To use this target, users would run python `run.py features` in the Terminal.

Build scripts also make the project development life much easier. Here’s an example workflow:

- Write ETL logic in `src/etl.py`.
- Import `etl` in `run.py`. Create a target called `data` that, when run via `python run.py data`, calls the relevant functions in etl to “build” the data.
- Work in notebooks to develop features, and, once no longer experimental, transfer feature creation code to `src/features.py`.
- Add a `feature` creation call to `run.py` under the target features, so `python run.py features` “builds” the features in the project. This should be done without rebuilding the data, if possible!

By following this workflow, it’ll make it easy to update different parts of our project when, say, the datasets change. An example of so would be something like:

```python
import sys
from src import etl, features

def main(target):
    if target == "data":
        etl.build_data()
    elif target == "features":
        features.build_features()
    else:
        print(f"Unknown target: {target}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [target]")
    else:
        main(sys.argv[1])
```

## Run.sh
On the other hand, we can setup bash script directly, which **automate system-level workflows** (execute shell commands, scripts, and pipelines) rather than Python-specific workflows (call library functions directly in code). This may involve OS-level tasks:

- installing dependencies (`conda env create`, `conda activate ...`),
- setting environment variables `MUJOCO_BACKEND_...`,
- copying files (`mkdir data`, `m -rf build/`, `curl https://...`),
- running multiple Python or R scripts (`src/train.py`).

An example for a graph transformer project running in the background on an DSMLP cluster would look like the following:

```bash
#!/usr/bin/env bash

set -Eeuo pipefail                        # Safer Bash: E=exit on error

# ---- Basics you might override per run ----
PY=python                                  # Which Python executable to use (override via env if needed).
DATA_DIR="${DATA_DIR:-./data}"             # Where raw/processed data lives; default ./data unless env set.
OUT_DIR="${OUT_DIR:-./outputs}"            # Where to write outputs (logs, artifacts, etc.).
CKPT_DIR="${CKPT_DIR:-$OUT_DIR/ckpt}"      # Where to save model checkpoints.
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"        # Where to save logs/TensorBoard files.

# Preferred GPU selection if on DSMLP; override per run
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # Pick GPU 0 by default; override like "0,1" for multi-GPU.

# --- Help text shown when no/invalid command is passed ---
usage() {
  cat <<EOF
Usage:
  ./run.sh data  [CONFIG]    # e.g. ./run.sh data configs/data.yaml
  ./run.sh train [CONFIG]    # e.g. ./run.sh train configs/train.yaml
  ./run.sh eval  [CONFIG]    # e.g. ./run.sh eval  configs/eval.yaml
  ./run.sh tb                # tensorboard on \$OUT_DIR
  ./run.sh env               # print paths/GPU info
Notes:
  - CONFIG defaults to a sane file if omitted (see below).
  - Override dirs with env vars: DATA_DIR, OUT_DIR, CKPT_DIR, LOG_DIR.
  - Set GPU with CUDA_VISIBLE_DEVICES (e.g., 0 or "0,1").
EOF
}

# --- Show environment/path details and GPU availability ---
cmd_env() {
  echo "DATA_DIR=$DATA_DIR"
  echo "OUT_DIR=$OUT_DIR"
  echo "CKPT_DIR=$CKPT_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  if command -v nvidia-smi >/dev/null 2>&1; then  # If NVIDIA tools exist, print GPU status.
    nvidia-smi || true
  else
    echo "(No GPU detected; running on CPU.)"
  fi
}

# --- Define the actual setups ---
cmd_data() {                               # Data preparation entrypoint (config-driven).
  CONFIG="${1:-configs/data.yaml}"         # Default config if none provided.
  mkdir -p "$DATA_DIR" "$OUT_DIR"          # Ensure directories exist.
  echo "→ Preparing data (config: $CONFIG)"
  $PY -m src.data.prepare \                # Run your data prep module as a package.
    --config "$CONFIG" \                   # Pass config path to your Python code.
    --data_dir "$DATA_DIR" \               # Where to read/write data.
    --out_dir "$OUT_DIR"                   # Where to store any derived artifacts/logs.
}

cmd_train() {                              # Training entrypoint (config-driven).
  CONFIG="${1:-configs/train.yaml}"        # Default training config.
  mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR" # Ensure output, checkpoint, log dirs exist.
  echo "→ Training (config: $CONFIG)"
  # Your train script should read all hyperparams/paths from the config
  $PY -m src.train \                       # Run your training module.
    --config "$CONFIG" \                   # Training config file.
    --data_dir "$DATA_DIR" \               # Data root.
    --out_dir "$OUT_DIR" \                 # General outputs (e.g., metrics, artifacts).
    --ckpt_dir "$CKPT_DIR" \               # Where to save checkpoints.
    --log_dir "$LOG_DIR"                   # Where to write logs/TensorBoard.
}

cmd_eval() {                               # Evaluation entrypoint (config-driven).
  CONFIG="${1:-configs/eval.yaml}"         # Default eval config.
  mkdir -p "$OUT_DIR"                      # Ensure outputs exist (for reports/plots).
  echo "→ Evaluating (config: $CONFIG)"
  $PY -m src.eval \                        # Run your eval module.
    --config "$CONFIG" \                   # Eval config file.
    --data_dir "$DATA_DIR" \               # Data root.
    --out_dir "$OUT_DIR" \                 # Where to put eval results.
    --ckpt_dir "$CKPT_DIR"                 # Where to load checkpoints from.
}

cmd_tb() {                                 # Start TensorBoard to browse logs.
  PORT="${TB_PORT:-6006}"                  # Default port 6006; override with TB_PORT=XXXX.
  echo "→ TensorBoard on $OUT_DIR (port $PORT)"
  tensorboard --logdir "$OUT_DIR" --port "$PORT"
}

# --- Running actual setups ---
CMD="${1:-}"; shift || true                # First CLI arg is the command; shift to pass remaining to handler.
case "$CMD" in
  env)   cmd_env "$@";;                    # ./run.sh env
  data)  cmd_data "$@";;                   # ./run.sh data [CONFIG]
  train) cmd_train "$@";;                  # ./run.sh train [CONFIG]
  eval)  cmd_eval "$@";;                   # ./run.sh eval [CONFIG]
  tb)    cmd_tb "$@";;                     # ./run.sh tb
  ""|-h|--help|help) usage;;               # ./run.sh (no args) or help → show usage
  *) echo "Unknown command: $CMD"; usage; exit 1;;  # Guard for typos/unknown commands.
esac
```
