# KataGo to Core ML - Quick Start

Minimal command sequence for converting KataGo models to Core ML. For detailed explanations, see [README.md](README.md).

## Prerequisites

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Miniconda (Apple Silicon)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

## Setup and Build

```bash
# Clone repository
mkdir -p ~/katago_workspace && cd ~/katago_workspace
git clone https://github.com/ChinChangYang/coremltools.git
cd coremltools
git checkout katagocoremltools

# Build (30-60 minutes)
make build

# Activate environment
source scripts/env_activate.sh --python=3.11

# Verify
python -c "from coremltools.converters import katago; print('OK')"
```

## Download Model

```bash
cd ~/katago_workspace
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
```

## Convert

```bash
# Create conversion script
cat > convert_katago.py << 'EOF'
#!/usr/bin/env python3
import sys, coremltools as ct

mlmodel = ct.converters.katago.convert(
    "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
    minimum_deployment_target=ct.target.iOS15
)
mlmodel.save("KataGo.mlpackage")
print("Conversion complete!")
EOF

# Run conversion (5-10 minutes)
python convert_katago.py
```

## Test

```bash
# Create test script
cat > test_inference.py << 'EOF'
#!/usr/bin/env python3
import numpy as np, coremltools as ct

model = ct.models.MLModel("KataGo.mlpackage")
result = model.predict({
    "spatial_input": np.random.randn(1, 22, 19, 19).astype(np.float32),
    "global_input": np.random.randn(1, 19).astype(np.float32),
    "input_mask": np.ones((1, 1, 19, 19), dtype=np.float32)
})
print("Inference test PASSED!")
for key, val in result.items():
    if hasattr(val, 'shape'):
        print(f"  {key}: {val.shape}")
EOF

# Run test
python test_inference.py
```

## Verification Checklist

```bash
conda --version                                                    # ✓ conda 24.x.x
python --version                                                   # ✓ Python 3.11.x
python -c "import coremltools; print(coremltools.__version__)"    # ✓ 9.0
python -c "from coremltools.converters import katago; print('OK')" # ✓ OK
ls kata1-*.bin.gz                                                  # ✓ ~271MB file
ls KataGo.mlpackage/                                               # ✓ Package directory
python test_inference.py                                           # ✓ Test passed
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `conda: command not found` | `source ~/.zshrc` |
| Build fails | `xcode-select --install` |
| `libcoremlpython` not found | `make clean && make build` |
| Conversion fails | Check model is v15 or v16 |
| Slow inference | Ensure using Apple Silicon |

## File Locations

```
~/katago_workspace/
├── coremltools/                          # Repository
│   ├── coremltools/converters/katago/    # Converter source
│   └── envs/KataGoCoremltools-py3.11/    # Conda environment
├── kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz  # Input model (~271MB)
├── KataGo.mlpackage/                     # Output model (~250-300MB)
├── convert_katago.py                     # Conversion script
└── test_inference.py                     # Test script
```

## Next Steps

- See [README.md](README.md) for detailed documentation
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common errors
- See [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md) for KataGo integration

## One-Liner Summary

```bash
# Complete setup and conversion (run after prerequisites)
cd ~/katago_workspace && \
git clone https://github.com/ChinChangYang/coremltools.git && \
cd coremltools && git checkout katagocoremltools && make build && \
source scripts/env_activate.sh --python=3.11 && cd ~/katago_workspace && \
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz && \
python -c "import coremltools as ct; ct.converters.katago.convert('kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz').save('KataGo.mlpackage')" && \
echo "Done! Model saved to KataGo.mlpackage"
```

---

**Total time**: ~60-90 minutes (mostly automated build and download)
