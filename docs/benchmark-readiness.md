# Benchmark Readiness

The current benchmark is a synthetic smoke / regression suite. It is not a real-world generalization benchmark. It exists to make policy, reward, report, and manifest changes repeatable while no real contract sample source is available.

## Groups

- `coverage_dominant`: a reachable candidate with stronger expected coverage can
  beat higher raw utility.
- `risk_dominant`: high-risk candidates exercise risk penalties and reporting.
- `path_cost_dominant`: expensive candidates exercise path-cost penalties.
- `sparse_candidates`: short candidate lists exercise padding masks.
- `high_unreachable_rate`: no reachable candidates exercise empty action masks.
- `missing_experimental_fields`: stable v1 fields still run with compatibility
  defaults when optional experimental fields are absent.

## Readiness Rules

- Generated scenarios must be deterministic for a fixed seed, group set,
  scenario count, and difficulty.
- Difficulty only changes candidate count and risk or cost scale; it must not
  change `model-explorer-contract/v1` stable field semantics.
- Synthetic benchmark results are valid for regression gates, not for claims
  about real lunar terrain performance.
