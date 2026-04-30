# Remote setup: binary PyTorch3D, build only the rasterizer

This is the lowest-friction setup for a remote machine with a newer system CUDA
installation, including CUDA 13. The project runs with an isolated conda CUDA
11.8 toolchain, installs PyTorch3D as a binary package, and locally builds only
`diff_gaussian_rasterization`.

## Why this stack

- The original project environment targets Python 3.9, PyTorch 2.0.1, CUDA 11.8.
- PyTorch3D 0.7.4 has a matching conda binary for this stack.
- `simple-knn` is no longer required because the project has a PyTorch fallback.
- `fused-ssim` is optional because training already falls back to pure PyTorch SSIM.
- `faiss-gpu` is optional because KMeans falls back to scikit-learn.
- `diff_gaussian_rasterization` is still required by the renderer and must match
  PyTorch/CUDA exactly.

## Clone

```bash
git clone --recursive <repo-url> GaussianPlant
cd GaussianPlant
git submodule sync --recursive
git submodule update --init --recursive
```

The renderer code in this branch expects the `dr_aa` rasterizer API with depth
and antialiasing support. The required checkout is:

```text
submodules/diff-gaussian-rasterization: 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
```

If the parent repository has not yet committed that gitlink, force the submodule
to the compatible commit:

```bash
git -C submodules/diff-gaussian-rasterization fetch origin dr_aa
git -C submodules/diff-gaussian-rasterization checkout 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
```

If `submodules/diff-gaussian-rasterization/third_party/glm` is empty:

```bash
git -C submodules/diff-gaussian-rasterization submodule update --init --recursive
```

## Create environment

```bash
conda env create -f environment-cu118-minimal.yml
conda activate gaussianplant-cu118
```

Force builds to use the conda CUDA compiler, not the system CUDA compiler:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Check that `nvcc` and PyTorch agree:

```bash
which nvcc
nvcc --version

python - <<'PY'
import torch
import pytorch3d
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("pytorch3d ok")
PY
```

Expected CUDA version: `11.8`.

## Build the only required local extension

```bash
pip install -v submodules/diff-gaussian-rasterization
```

Do not install these source submodules in this minimal setup:

```bash
# not needed
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

## Verify project imports

```bash
python - <<'PY'
import torch
import pytorch3d
import diff_gaussian_rasterization
from scene.gaussian_model import distCUDA2

print("torch", torch.__version__, torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("pytorch3d ok")
print("diff_gaussian_rasterization ok")
print("distCUDA2", distCUDA2)
PY
```

If the build fails with a CUDA mismatch, the wrong `nvcc` is first on `PATH`.
Re-export `CUDA_HOME`/`PATH`, remove the failed build directory, and reinstall:

```bash
rm -rf submodules/diff-gaussian-rasterization/build
pip install -v --no-cache-dir --force-reinstall submodules/diff-gaussian-rasterization
```
