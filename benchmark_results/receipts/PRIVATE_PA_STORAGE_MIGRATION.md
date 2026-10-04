# Private PA Storage Migration Receipt

- **Date:** 2026-10-04
- **Status:** COMPLETE, with GitHub private-remote secrets still requiring configuration before the next secure cloud replay
- **Package:** `pa-model-locked-2026-reproducibility-package.zip`
- **Size:** `50,369,372` bytes
- **SHA-256:** `a75b982f8e814cd427fc553e798a7b8533a46fcf7c9f880cd26d8a3dde722c95`

## What changed

- Copied the locked PA reproducibility package to access-controlled private storage.
- Verified the private copy against the original package digest.
- Deleted public GitHub Actions artifact `11287257159` and verified that the originating run no longer returns that artifact.
- Deleted raw Statcast Actions caches with keys:
  - `statcast-pa-v1-2023-03-20-2025-11-05`
  - `statcast-pa-locked-2026-v1-2023-03-20-2026-10-03`
- Removed or replaced public-artifact replay workflows.
- Added a fail-closed private restore script that verifies the package SHA before extraction.
- Updated `.gitignore` and repository documentation to prohibit bulk source-row redistribution.

## Cloud-run requirement

Before the next secure GitHub Actions replay, configure an authenticated private remote through:

- `BRL_RCLONE_CONFIG_B64`
- `BRL_PRIVATE_PA_REMOTE`

Local/private runners may instead set `BRL_PRIVATE_PA_PACKAGE_PATH`.
