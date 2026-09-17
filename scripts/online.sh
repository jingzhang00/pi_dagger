#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${OPENPI_PYTHON:-${WORKSPACE}/.venv/bin/python}"
CONFIG="${CONFIG:-${WORKSPACE}/dagger/config/config.yaml}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "OPENPI_PYTHON is not executable: ${PYTHON}" >&2
    exit 1
fi

cd "${WORKSPACE}"
export CONFIG
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/openpi/src:${WORKSPACE}/openpi/packages/openpi-client/src:${PYTHONPATH:-}"

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${WORKSPACE}/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a
    source "${WANDB_ENV_FILE}"
    set +a
fi

exec "${PYTHON}" -m dagger.online_runner --config "${CONFIG}" "$@"
