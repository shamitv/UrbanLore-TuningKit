# PyTorch on RTX 5080 (Windows) — Setup & Verification Guide

This guide is for **Windows + NVIDIA GeForce RTX 5080** and assumes your GPU is visible in `nvidia-smi` (you showed **Driver 591.44** on Windows/WDDM).

---

## 0) What matters for RTX 5080

RTX 5080 is a **Blackwell** GPU. To ensure PyTorch actually uses it, you want a PyTorch build that includes **Blackwell (sm_120)** support.

On Windows, the simplest reliable path is installing the **CUDA 12.8 (“cu128”) PyTorch wheels** from the official PyTorch wheel index.

> Note: The `CUDA Version` shown by `nvidia-smi` is what your **driver supports**. PyTorch uses its own packaged CUDA runtime (from the wheel), so you don’t need to install a full CUDA Toolkit for basic PyTorch usage.

---

## 1) Prerequisites (Windows)

- **NVIDIA Driver**: Already installed (verified via `nvidia-smi`).
- **Python**: Recommended **Python 3.12** (3.10+ also works).
- (Optional) **Git** / **VS Code** / etc.

---

## 2) Install PyTorch (recommended: virtual environment)

### PowerShell

```powershell
# Create a fresh project folder (optional)
mkdir pytorch-rtx5080
cd pytorch-rtx5080

# Create & activate a virtual env
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install -U pip

# Install PyTorch + CUDA 12.8 wheels (cu128)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CMD (Command Prompt)

```bat
mkdir pytorch-rtx5080
cd pytorch-rtx5080

py -3.12 -m venv .venv
.venv\Scripts\activate

python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 3) Verify PyTorch is using the GPU

Create a file **`verify_gpu.py`**:

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda runtime:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # expect (12, 0) on RTX 5080
```

Run it:

```powershell
python verify_gpu.py
```

Expected highlights:

- `cuda available: True`
- `gpu: NVIDIA GeForce RTX 5080`
- `capability: (12, 0)` *(the key confirmation for Blackwell / sm_120)*

---

## 4) Quick performance sanity test

Create **`matmul_test.py`**:

```python
import torch, time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

if device == "cuda":
    torch.cuda.synchronize()

t0 = time.time()
c = a @ b

if device == "cuda":
    torch.cuda.synchronize()

print("secs:", time.time() - t0)
print("mean:", c.mean().item())
```

Run:

```powershell
python matmul_test.py
```

While running, you can watch GPU usage:

```powershell
nvidia-smi -l 1
```

---

## 5) Common problems & fixes

### A) `torch.cuda.is_available()` is `False`

1. Make sure you installed from the **cu128** index URL:

   ```powershell
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

2. Close/reopen your terminal and re-activate `.venv`.

3. Confirm Windows sees the GPU:

   ```powershell
   nvidia-smi
   ```

### B) Warnings about unsupported architecture / falling back to CPU

This usually happens if you installed a CPU-only wheel or an older CUDA wheel.
Reinstall with **cu128** as shown above.

### C) You installed something else and PyTorch got downgraded

Check:

```powershell
pip show torch
pip list | findstr torch
```

Then reinstall from the cu128 index.

---

## 6) Tips for better speed

### Use mixed precision
For many deep learning workloads, automatic mixed precision speeds things up:

```python
import torch
from torch import nn

model = nn.Linear(4096, 4096).cuda()
x = torch.randn(1024, 4096, device="cuda")

with torch.autocast(device_type="cuda", dtype=torch.float16):
    y = model(x)
```

### Enable TF32 for matmul (often faster, minimal impact)
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 7) If you also use WSL2

- Install the NVIDIA driver on Windows (already done).
- Inside WSL, install **the same cu128 PyTorch wheels** in your Linux venv.
- Confirm `nvidia-smi` works inside WSL.

---

## 8) Minimal template you can copy into any project

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

x = torch.randn(1024, 1024, device=device)
y = x @ x

if device.type == "cuda":
    torch.cuda.synchronize()

print(y.mean().item())
```

---

### Done ✅

If you paste the output of `verify_gpu.py`, you can confirm in one glance whether you’re on the correct CUDA wheel and seeing **sm_120** properly.
