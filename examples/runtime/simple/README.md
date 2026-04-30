# Simple Distributed Finance Runtime Examples

These are small, copyable distributed runtime starting points for finance
workflows.

- `distributed_publish_fan_in.py`: one planner publishes a portfolio review to
  market-data and risk workers, and one stateful aggregator collects both
  results by `job_id`.
- `distributed_scatter_gather.py`: one coordinator sends direct RPCs to
  execution and risk specialists and merges both responses into one result.

## Run

```bash
uv run python examples/runtime/simple/distributed_publish_fan_in.py
uv run python examples/runtime/simple/distributed_scatter_gather.py
```
