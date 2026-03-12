# TorchDynamo and torch.compile Fixes

## Issue
Training runs using PyTorch 2.0+ `torch.compile` (`TorchDynamo`) crashed with a `SpeculationLog diverged` error. This was caused by the `WhiteningPenalty` module in the Zipformer scaling codebase (`models/zipformer_scaling.py`).

The original implementation contained stochastic control flow within the `forward` pass to probabilistically apply the expensive whitening penalty calculation:

```python
# The original offending line
if not x.requires_grad or torch.rand(1).item() > self.prob or grad_scale == 0:
    return _no_op(x)
```

TorchDynamo relies on static, repeatable computational graphs. When it traces the `forward` pass, random path execution causes the graph speculation to diverge on subsequent passes, crashing the compiler.

## Solution
To solve this and allow seamless integration with `torch.compile` speedups, the randomness was removed from the strictly-traced `forward` pass and pushed into the dynamically-executed `backward` pass of the custom `torch.autograd.Function`.

**1. Updated `models/zipformer_scaling.py` (WhiteningPenalty `forward`):**
```python
grad_scale = float(self.grad_scale)
if not x.requires_grad or grad_scale == 0:
    return _no_op(x)
```
*Dynamo now correctly compiles this as a static, predictable graph path.*

**2. Updated `models/zipformer_scaling.py` (WhiteningPenaltyFunction `backward`):**
```python
@staticmethod
def backward(ctx, x_grad: Tensor):
    (x_orig,) = ctx.saved_tensors
    w = ctx.module

    # Stochastic execution shifted here
    if random.random() > w.prob:
        return x_grad, None
```
*Because custom `autograd.Function` backward methods are executed eagerly/dynamically during the backward pass rather than being traced strictly by Dynamo, the stochastic skipping logic executes correctly without crashing the graph speculation.*
