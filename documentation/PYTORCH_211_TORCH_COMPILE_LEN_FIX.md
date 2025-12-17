# PyTorch 2.11+ torch.compile `len()` Compatibility Fix

## Problem Summary

When using PyTorch 2.11.0a0 (nightly) or later with `torch.compile` enabled (`dynamo_backend: "inductor"`), validation fails with:

```
Flux2Transformer2DModel does not support len()
```

This error occurs during validation inference and prevents monitoring for overfitting/model collapse during training.

## Root Cause

PyTorch 2.11 introduced a behavioral change in `torch._dynamo.eval_frame.OptimizedModule` (the wrapper class used by `torch.compile`).

### Before PyTorch 2.11

```python
# OptimizedModule did NOT have __len__
hasattr(compiled_model, '__len__')  # Returns False
len(compiled_model)  # Raises: TypeError: object of type 'OptimizedModule' has no len()
```

### PyTorch 2.11+

```python
# OptimizedModule now HAS __len__ that delegates to the wrapped model
def __len__(self) -> int:
    if isinstance(self._orig_mod, Sized):
        return len(self._orig_mod)
    raise TypeError(f"{type(self._orig_mod).__name__} does not support len()")

hasattr(compiled_model, '__len__')  # Returns True
len(compiled_model)  # Tries to call len() on the wrapped model
```

This means code that checks `hasattr(model, '__len__')` before calling `len()` will now attempt to call `len()` on compiled models, which then fails if the underlying model doesn't implement `__len__`.

## Affected Configurations

- **PyTorch Version**: 2.11.0a0+ (nightly builds from late 2025)
- **Training Mode**: Any configuration using `torch.compile`:
  ```json
  {
    "dynamo_backend": "inductor",
    "dynamo_mode": "max-autotune-no-cudagraphs"
  }
  ```
- **Model**: Flux2Transformer2DModel (and potentially other transformer models)

## Solution

Add a `__len__` method to the transformer model class that returns a sensible value.

### Implementation

File: `simpletuner/helpers/models/flux2/transformer.py`

```python
class Flux2Transformer2DModel(...):
    # ... existing code ...

    def __len__(self) -> int:
        """
        Return the total number of transformer blocks.

        This is needed for compatibility with PyTorch 2.11+ where torch.compile's
        OptimizedModule delegates __len__ to the wrapped model.
        """
        return len(self.transformer_blocks) + len(self.single_transformer_blocks)
```

For Flux2, this returns 56 (8 double-stream blocks + 48 single-stream blocks).

### Location of Fix

The fix was added after the `__init__` method, around line 865:

```python
        # TREAD router for efficient training
        self._tread_router = None
        self._tread_routes = None

    def __len__(self) -> int:
        """
        Return the total number of transformer blocks.

        This is needed for compatibility with PyTorch 2.11+ where torch.compile's
        OptimizedModule delegates __len__ to the wrapped model.
        """
        return len(self.transformer_blocks) + len(self.single_transformer_blocks)

    def set_router(self, router, routes: List[Dict[str, Any]]):
```

## Verification

Test that the fix works:

```python
import torch
from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel

# Verify __len__ is defined
print('Has __len__:', hasattr(Flux2Transformer2DModel, '__len__'))

# Test with a compiled model
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(8)])
        self.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(48)])

    def __len__(self):
        return len(self.transformer_blocks) + len(self.single_transformer_blocks)

    def forward(self, x):
        return x

model = TestModel()
compiled = torch.compile(model)
print(f'len(compiled): {len(compiled)}')  # Should print 56
```

## Alternative Workarounds

If you cannot modify the transformer code:

### Option 1: Disable torch.compile

```json
{
  "dynamo_backend": "none"
}
```

**Trade-off**: Loses torch.compile performance benefits (typically 10-30% slower training).

### Option 2: Use stable PyTorch

Downgrade to PyTorch 2.10.x or earlier where `OptimizedModule` doesn't have `__len__`.

**Trade-off**: May lose access to newer features and optimizations, especially for cutting-edge hardware like B300 GPUs.

### Option 3: Patch OptimizedModule at runtime

```python
# Add to your training script before any torch.compile calls
from torch._dynamo.eval_frame import OptimizedModule

def safe_len(self):
    # Don't delegate, just raise the old-style error
    raise TypeError("object of type 'OptimizedModule' has no len()")

OptimizedModule.__len__ = safe_len
```

**Trade-off**: May break other code that legitimately expects `len()` to work on compiled models.

## Related Information

- **PyTorch Version Tested**: 2.11.0a0+cuV13088.b300.sdpafix
- **Hardware**: NVIDIA B300 SXM6 AC (SM103, Blackwell architecture)
- **CUDA**: 13.0
- **Training Framework**: SimpleTuner with LyCORIS (LoKr + DoRA)

## Applying to Other Models

If you encounter this error with other transformer models, add a similar `__len__` method:

```python
def __len__(self) -> int:
    """Return a sensible length for the model."""
    # For models with transformer blocks:
    if hasattr(self, 'transformer_blocks'):
        return len(self.transformer_blocks)
    # For models with layers:
    if hasattr(self, 'layers'):
        return len(self.layers)
    # Fallback: count all child modules
    return len(list(self.children()))
```

---

**Document Created**: 2025-12-17
**Issue**: Validation fails with "does not support len()" on PyTorch 2.11+ with torch.compile
**Resolution**: Add `__len__` method to Flux2Transformer2DModel
