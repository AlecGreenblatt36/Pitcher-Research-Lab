# Private Data Boundary

Bulk MLB/Statcast rows are not distributed through this public repository or public GitHub Actions storage.

## Locked private package

- File: `pa-model-locked-2026-reproducibility-package.zip`
- Expected size: `50,369,372` bytes
- SHA-256: `a75b982f8e814cd427fc553e798a7b8533a46fcf7c9f880cd26d8a3dde722c95`
- Storage: approved private external storage; access-controlled and not anonymously shared

The public repository stores only code, compact receipts, hashes, aggregate benchmark results, and documentation.

## Restore methods

`scripts/restore_private_pa_package.sh` fails closed and supports:

1. `BRL_PRIVATE_PA_PACKAGE_PATH` for a local/private runner; or
2. an authenticated rclone remote configured through the GitHub secrets:
   - `BRL_RCLONE_CONFIG_B64`
   - `BRL_PRIVATE_PA_REMOTE`

The script verifies the package SHA-256 before extracting it.

## Prohibited public storage

Do not upload or cache any of the following in public GitHub Actions:

- `plate_appearances.csv.gz` or raw Statcast rows;
- row-level locked PA predictions;
- full reproducibility ZIPs containing bulk source rows;
- public Actions caches whose payload contains bulk Statcast data.

Derived game-level forecast rows may be retained only when they contain no bulk source rows and the relevant data terms permit the intended use.
