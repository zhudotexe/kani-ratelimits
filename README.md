# kani-ratelimits

This is a simple, small package to to enforce request-per-minute (RPM), token-per-minute (TPM), and/or max-concurrency
ratelimits before making requests to an underlying engine.

## Installation

```shell
pip install kani-ratelimits
```

## Usage

```python
from kani.ext.ratelimits import RatelimitedEngine

# limit requests to 10 req/min and 30k tokens/min
inner_engine = ...  # your engine here, e.g. `OpenAIEngine(..., model="gpt-4")`
engine = RatelimitedEngine(inner_engine, rpm_limit=10, tpm_limit=30_000)
```

The `RatelimitedEngine` takes the following parameters:

- `engine`: The engine to wrap.
- `max_concurrency` (int): The maximum number of concurrent requests to serve at once (default unlimited).
- `rpm_limit` (float): The maximum number of requests to serve per *rpm_period* (default unlimited).
- `rpm_period` (float): The duration, in seconds, of the time period in which to limit the rate. Note that up to
  *rpm_limit* requests are allowed within this time period in a burst (default 60s).
- `tpm_limit` (float): The maximum number of tokens to send in requests per *tpm_period* (default unlimited).
- `tpm_period` (float): The duration, in seconds, of the time period in which to limit the rate. Note that up to
  *tpm_limit* tokens are allowed within this time period in a burst (default 60s).

The ratelimiter will ensure that all conditions are met before forwarding the request to the wrapped engine.