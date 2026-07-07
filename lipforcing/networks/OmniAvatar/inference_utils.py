"""torch.compile helpers providing a `conditional_compile` decorator.

Why per-function decoration instead of `torch.compile(model)`?
- Whole-module compile of the AR forward triggers a Dynamo recompile per
  unique combination of guard variables (cur_start_frame, store_kv,
  t_cur, KV cache state). With ~50+ combinations across an inference
  it exceeds the Dynamo cache limit and ends up retracing per call,
  regressing throughput by 50-60x.
- Decorating individual hot functions (rope_apply, _forward_ar) lets
  Dynamo fuse op-level kernels without entangling guard variables that
  belong to outer Python control flow.

Set `LIPFORCING_COMPILE=true` in the environment BEFORE the modules using
@conditional_compile are first imported. Default off.
"""
import os
import torch

ENABLE_COMPILE = os.getenv("LIPFORCING_COMPILE", "false").lower() == "true"
print(f"[inference_utils] LIPFORCING_COMPILE={ENABLE_COMPILE}")
torch._dynamo.config.cache_size_limit = 128


def conditional_compile(func):
    """Decorator: torch.compile the function iff LIPFORCING_COMPILE=true."""
    if ENABLE_COMPILE:
        return torch.compile(mode=None, backend="inductor", dynamic=None)(func)
    return func
