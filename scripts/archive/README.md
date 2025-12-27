# Archived Optimization Experiments

This directory contains scripts from historical optimization experiments that were conducted to find the best configuration for the KataGo Core ML converter.

## Archived Scripts

### `run_phase4_experiments.py`

**Purpose**: Automated test runner for Phase 4 optimization experiments (low-priority optimizations).

**Experiments tested**:
1. Skip irrelevant optimization passes (GELU, PReLU, LayerNorm fusion)
2. Aggressive const elimination (10MB threshold)
3. Optimized fusion pass order

**Results**: All experiments showed no meaningful performance impact (within measurement noise of ±1%). See `../optimization_results.json` for complete data.

**Decision**: Custom pass pipelines were removed from the converter API. The default Core ML pass pipeline is already well-optimized for KataGo models.

## Why These Scripts Are Archived

These scripts represent experiments that:
- Did not provide meaningful performance improvements
- Added code complexity without sufficient benefit
- Were superseded by better optimizations (iOS18 target, identity mask elimination)

They are preserved here as historical record of what was tried and why certain approaches were rejected. This helps future contributors understand the optimization exploration process and avoid re-testing known dead ends.

## Current Recommendations

For current optimization recommendations and active converter parameters, see:
- **User documentation**: `../../docs/katago/README.md` (Performance Optimization section)
- **Technical details**: `../../docs/optimization_summary.md`
- **Complete experimental data**: `../optimization_results.json`

## Key Findings

**Best performing optimizations** (kept in codebase):
1. **Identity mask elimination**: 6.5% speedup (for full 19x19 boards)
2. **iOS18 deployment target**: 1.4% speedup (backward compatible)

**Rejected optimizations** (archived or removed):
1. Mish activation variants (softplus/SiLU): Failed validation or slower
2. Fused linear operations: Only 0.7% speedup, added complexity
3. Custom pass pipelines: No measurable impact

---

**Archive Date**: 2025-12-27
**Reason**: Code cleanup after optimization experiments completion
