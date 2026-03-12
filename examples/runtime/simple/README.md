# Simple Distributed Runtime Examples

These are small, copyable distributed runtime starting points.

- `distributed_publish_fan_in.py`: one planner publishes work to two specialist
  workers, and one stateful aggregator collects both results by `job_id`.
- `distributed_scatter_gather.py`: one coordinator sends direct RPCs to two
  specialist workers and merges both responses into one result.

## Run

```bash
uv run python examples/runtime/simple/distributed_publish_fan_in.py
uv run python examples/runtime/simple/distributed_scatter_gather.py
```
