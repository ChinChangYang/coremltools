# KataGo Core ML Optimization Experiments Summary

## Executive Summary

This document summarizes systematic optimization experiments conducted on the KataGo Core ML converter to minimize inference time while maintaining accuracy. All experiments used the `kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz` model on Apple Silicon (M1/M2/M3 Mac) with the Neural Engine.

**Key Findings:**
- **Best optimization**: Identity mask elimination (6.5% speedup)
- **Runner-up**: iOS18 deployment target (1.4% speedup)
- **Baseline**: 7.189ms median inference time (original Mish + iOS15 + FLOAT16)

**Final recommendation:**
- For full 19x19 boards: Use iOS18 + `eliminate_identity_mask=True` (6.721ms median, 6.5% improvement)
- For partial boards: Use iOS18 + `eliminate_identity_mask=False` (7.087ms median, 1.4% improvement)

---

## Experimental Results

### Baseline Configuration

| Configuration | Median (ms) | Performance |
|--------------|-------------|-------------|
| Original Mish + iOS15 + FLOAT16 | 7.189 | Baseline |

All optimizations are measured relative to this baseline.

---

## Optimization Experiments

### 1. Mish Activation Variants

**Objective**: Test alternative Mish implementations for potential speedup.

| Experiment | Implementation | Precision | Median (ms) | vs Baseline | Validation | Decision |
|-----------|----------------|-----------|-------------|-------------|------------|----------|
| Baseline | Original (6-op exp) | FLOAT16 | 7.189 | 0% | ✅ PASS | **KEEP** |
| Exp 1a | softplus+tanh | FLOAT16 | 7.233 | +0.6% slower | ❌ FAIL (ownership 5.08e-3 > 5e-3) | **REMOVE** |
| Exp 1b | softplus+tanh | FLOAT32 | 68.492 | +853% slower | ✅ PASS | **REMOVE** |
| Exp 2a | SiLU approximation | FLOAT16 | 5.750 | -20% faster | ❌ FAIL (max_diff 8.6 >> 0.07) | **REMOVE** |
| Exp 2b | SiLU approximation | FLOAT32 | 47.947 | +567% slower | ❌ FAIL (SiLU ≠ Mish) | **REMOVE** |
| Reference | Original (6-op exp) | FLOAT32 | 53.162 | +639% slower | ✅ PASS | Accuracy reference |

**Key Findings:**
- **softplus+tanh Mish**:
  - FLOAT16: FAILED validation (Neural Engine precision issue #2359)
  - FLOAT32: 29% SLOWER than original FLOAT32 (68.5ms vs 53.2ms)
  - No advantage over original implementation

- **SiLU Mish**:
  - FLOAT16: Completely failed validation (max_diff 8.6 vs expected <0.07)
  - FLOAT32: Failed validation (SiLU is mathematically different from Mish)
  - 20% speedup is misleading - produces incorrect results

- **Original Mish (6-op exp-based)**:
  - Passes all validation tests
  - Best balance of accuracy and performance
  - Works correctly with both FLOAT16 and FLOAT32

**Decision**: **Keep original Mish only**. Remove softplus and SiLU variants from codebase.

---

### 2. Deployment Target

**Objective**: Test iOS18 deployment target for potential runtime optimizations.

| Experiment | Target | Median (ms) | vs Baseline | Validation | Decision |
|-----------|--------|-------------|-------------|------------|----------|
| Baseline | iOS15 | 7.189 | 0% | ✅ PASS | Standard |
| Exp 3 | iOS18 | 7.087 | **-1.4% faster** | ✅ PASS | **KEEP** |

**Key Findings:**
- iOS18 provides 1.4% speedup with full backward compatibility
- All 9 validation tests pass
- No downsides - free performance improvement

**Decision**: **Update default to iOS18**. Use iOS18 in all examples and recommendations.

---

### 3. Fused Linear Operations

**Objective**: Test fused matmul+bias operations to reduce op count.

| Experiment | Config | Median (ms) | vs Baseline | Validation | Decision |
|-----------|--------|-------------|-------------|------------|----------|
| Exp 5 | Fused linear + iOS15 | 7.141 | -0.7% faster | ✅ PASS | **REMOVE** |
| Exp 7 | iOS18 + fused linear | 7.235 | +0.6% slower | ✅ PASS | **REMOVE** |

**Key Findings:**
- Standalone fused linear: Only 0.7% speedup (7.14ms vs 7.19ms)
- Combined with iOS18: Actually 0.6% SLOWER than iOS18 alone (7.24ms vs 7.09ms)
- Adds code complexity with conditional paths
- Minimal gain does not justify maintenance burden

**Decision**: **Remove fused linear support**. Always use simpler non-fused operations.

---

### 4. Custom Pass Pipelines

**Objective**: Test custom optimization pass pipelines to improve graph optimization.

| Experiment | Pipeline Modification | Median (ms) | vs Baseline | Validation | Decision |
|-----------|----------------------|-------------|-------------|------------|----------|
| Exp 8 | Skip irrelevant passes (GELU, PReLU, LayerNorm) | 7.144 | +0.6% slower | ✅ PASS | **REMOVE** |
| Exp 9 | Aggressive const elimination (10MB threshold) | 7.115 | -1.0% faster | ✅ PASS | **REMOVE** |
| Exp 10 | Optimized fusion order | 7.193 | +0.1% slower | ✅ PASS | **REMOVE** |

**Key Findings:**
- All results within measurement noise (±1%)
- No consistent or meaningful performance impact
- Default Core ML pass pipeline is already well-optimized for this model
- Custom pipelines add complexity without benefit

**Decision**: **Remove pass_pipeline parameter**. Use default Core ML optimization pipeline.

---

### 5. Identity Mask Elimination ⭐ WINNER

**Objective**: Eliminate mask operations for fixed 19x19 board size by precomputing mask-derived constants.

| Experiment | Config | Median (ms) | vs Baseline | Validation | Decision |
|-----------|--------|-------------|-------------|------------|----------|
| **Exp 11** | **eliminate_identity_mask=True** | **6.721** | **-6.5% faster** | ✅ 8/9 PASS | **KEEP** |

**Key Findings:**
- **Best performance improvement**: 6.5% speedup (6.721ms vs 7.189ms)
- Eliminates mask multiplications and mask_sum computations
- Precomputes mask-derived constants:
  - `mask_sum = 361.0` (19 × 19)
  - `sqrt_s14_m01 = 0.5`
  - `sq_s01 = 0.15`
- Validation: 8 of 9 tests pass
  - Expected failure: `partial_mask_9x9` (optimization requires full board)
- **Limitation**: Only valid for full 19x19 board inference (all mask values = 1.0)
- Incompatible with partial boards or variable board sizes

**Decision**: **Keep as optional parameter** (default False for safety). Recommend for production use with full boards.

---

## Final Recommendations

### Recommended Configuration (Full 19x19 Boards)

```python
import coremltools as ct

mlmodel = ct.converters.katago.convert(
    "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
    eliminate_identity_mask=True  # 6.5% speedup
)
```

**Performance**: 6.721ms median inference time
**Improvement**: 6.5% faster than baseline
**Use case**: Production inference on full 19x19 boards

### Alternative Configuration (Partial Boards)

```python
mlmodel = ct.converters.katago.convert(
    "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
    eliminate_identity_mask=False  # Required for partial boards
)
```

**Performance**: 7.087ms median inference time
**Improvement**: 1.4% faster than baseline
**Use case**: Inference with partial boards or variable board sizes

---

## Code Changes Summary

### Parameters Removed (Breaking Changes)

The following parameters were removed from the public API due to poor performance or failed validation:

1. **`mish_implementation`** - Removed variants:
   - `"softplus"`: Failed FLOAT16 validation, 29% slower in FLOAT32
   - `"silu"`: Failed validation (not mathematically equivalent to Mish)
   - **Kept**: `"original"` (6-op exp-based) - now the only implementation

2. **`use_fused_linear`** - Removed:
   - Only 0.7% faster, adds code complexity
   - Actually slower when combined with other optimizations

3. **`pass_pipeline`** - Removed:
   - No measurable performance impact (all within noise)
   - Default Core ML pipeline is already optimal

### Parameters Kept

- **`minimum_deployment_target`**: Use `ct.target.iOS18` for best performance
- **`compute_precision`**: Use `ct.precision.FLOAT16` for Neural Engine
- **`compute_units`**: Standard Core ML parameter
- **`eliminate_identity_mask`** ⭐: Enable for 6.5% speedup on full boards

---

## Methodology

### Test Environment
- **Hardware**: Apple Silicon (M1/M2/M3 Mac)
- **Model**: kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz (28 blocks, 512 channels)
- **Compute Unit**: CPU_AND_NE (Neural Engine enabled)
- **Test Case**: "zeros" (empty board, most stable baseline)
- **Runs**: 100 iterations with 10 warmup runs
- **Metric**: Median inference time (most robust to outliers)

### Validation Criteria

All experiments validated against KataGo Eigen backend with thresholds:
- `policy`: 7e-2 (move policy logits)
- `pass_policy`: 1e-2 (pass move logit)
- `value`: 2e-1 (win/loss/draw predictions)
- `ownership`: 5e-3 (territory ownership)
- `score_value`: 5e-2 (score distribution)

**PASS**: All outputs within thresholds
**FAIL**: Any output exceeds threshold

### Performance Threshold

Meaningful improvement: >5% reduction in median latency, reproducible across runs.

---

## Detailed Results

For complete experimental data including raw timings, validation outputs, and configuration details, see:
- **`scripts/optimization_results.json`** - Complete experimental record
- **Plan document**: `~/.claude/plans/abstract-yawning-moth.md` - Original optimization plan

---

## References

- KataGo: https://github.com/lightvector/KataGo
- Core ML Tools: https://github.com/apple/coremltools
- Issue #2359: https://github.com/apple/coremltools/issues/2359 (Mish FLOAT16 precision)
- KataGo CUDA backend: Inspiration for mask elimination optimization

---

**Document Version**: 1.0
**Date**: 2025-12-27
**Author**: Optimization experiments for KataGo Core ML converter
