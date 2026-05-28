# Contract Regression Fixtures

No real samples are available yet. The files in this directory are compatibility
fixtures only; they keep the JSON loading and missing-field fallback paths under
test until exported contracts from the upstream modeling stack exist.

Rules for future intake:

- Store contracts as plain JSON `model-explorer-contract/v1` files.
- Use descriptive names such as `real-YYYYMMDD-source-case.json` only for actual
  exported real samples; use `curated-synthetic-*` for hand-built fixtures.
- redact mission identifiers, operator names, private paths, and precise source
  metadata before adding a file.
- do not import `dev-platform-constraints`, `a_gcs_ws-2.0.1`, or any external
  project from these tests; consume fixtures only through the public JSON loader.
