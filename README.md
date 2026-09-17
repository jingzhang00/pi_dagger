# D1 PI DAgger


## Environment

```bash
cd pi_dagger/
git submodule update --init openpi
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project "$PWD/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PWD/.venv/bin/python" -e "$PWD/openpi"
uv pip install --python "$PWD/.venv/bin/python" "pin==2.7.0" pygame-ce
export PYTHONPATH="$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
```

## Step 0: Finetune the VLA on π0.5
```bash
source .venv/bin/activate
bash scripts/finetune.sh
```
- Training parameters, the dataset, the base model, and the output directory are read from the `vla` section of `dagger/config/config.yaml`.
- OpenPI automatically saves checkpoints by step under `vla.output/config_name/exp_name/`.
- Robot hardware and runtime parameters are read from `dagger/config/robot.yaml`.





## Port
```bash
ls -l /dev/serial/by-id/ 2>/dev/null
```