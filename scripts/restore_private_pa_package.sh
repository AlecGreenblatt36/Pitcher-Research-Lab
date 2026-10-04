#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA256="a75b982f8e814cd427fc553e798a7b8533a46fcf7c9f880cd26d8a3dde722c95"
DEST_DIR="${BRL_PRIVATE_PA_DEST_DIR:-pa_model_reference}"
ZIP_PATH="${RUNNER_TEMP:-/tmp}/brl-private-pa-package.zip"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"

if [[ -n "${BRL_PRIVATE_PA_PACKAGE_PATH:-}" ]]; then
  cp "${BRL_PRIVATE_PA_PACKAGE_PATH}" "${ZIP_PATH}"
elif [[ -n "${BRL_RCLONE_CONFIG_B64:-}" && -n "${BRL_PRIVATE_PA_REMOTE:-}" ]]; then
  command -v rclone >/dev/null 2>&1 || {
    curl -fsSL https://rclone.org/install.sh | sudo bash
  }
  mkdir -p "${HOME}/.config/rclone"
  printf '%s' "${BRL_RCLONE_CONFIG_B64}" | base64 --decode > "${HOME}/.config/rclone/rclone.conf"
  chmod 600 "${HOME}/.config/rclone/rclone.conf"
  rclone copyto "${BRL_PRIVATE_PA_REMOTE}" "${ZIP_PATH}" --config "${HOME}/.config/rclone/rclone.conf"
else
  echo "Private PA package unavailable." >&2
  echo "Set BRL_PRIVATE_PA_PACKAGE_PATH for a local run, or configure the GitHub secrets" >&2
  echo "BRL_RCLONE_CONFIG_B64 and BRL_PRIVATE_PA_REMOTE for an authenticated private remote." >&2
  exit 78
fi

actual_sha=$(sha256sum "${ZIP_PATH}" | awk '{print $1}')
if [[ "${actual_sha}" != "${EXPECTED_SHA256}" ]]; then
  echo "Private PA package hash mismatch: ${actual_sha}" >&2
  exit 1
fi

unzip -q "${ZIP_PATH}" -d "${DEST_DIR}"
test -f "${DEST_DIR}/model_runs/pa_locked_2026/plate_appearances.csv.gz"
test -f "${DEST_DIR}/model_runs/pa_locked_2026/artifacts/pa_model.joblib"
test -f "${DEST_DIR}/model_runs/pa_locked_2026/artifacts/test_predictions.csv.gz"

echo "Private PA package restored and verified: ${actual_sha}"
