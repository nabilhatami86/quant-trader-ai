# Architecture Status

This repository now uses `app/` as the primary application package:

- `app/api`
- `app/core`
- `app/database`
- `app/engine`
- `app/services`
- `app/utils`

Current state:

- Primary runtime imports point to `app.*`
- Core implementation files have been physically moved into `app/`
- Legacy top-level packages remain as compatibility shims
- File-based runtime storage has been normalized into:
  - `data/cache`
  - `data/journal`
  - `data/session`

Key rules:

1. New code should import from `app.*`
2. Legacy packages (`api`, `backend`, `db`, `core`, `services`, `utils`, `backtest`) should be treated as wrappers
3. Runtime paths should come from `app.core.paths` when possible

Suggested next phase:

1. Expand tests for `app.*` runtime flows
2. Replace remaining ad-hoc filesystem paths with `app.core.paths`
3. Remove legacy wrappers after a full test pass and deployment verification
