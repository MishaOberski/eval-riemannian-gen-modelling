# How Geometry Shapes Generative Model Evaluation on Riemannian Manifolds

This repository accompanies the paper *Reliably Evaluating Riemannian Generative Modelling*. The paper studies how the reliability of generative-model evaluation metrics (Wasserstein/Sinkhorn, MMD, and Fréchet-variance dispersion) for Riemannian Flow Matching (RFM) depends on curvature, dimensionality, and target-distribution structure, comparing intrinsic (geodesic), tangent-space, and ambient formulations of these metrics through controlled synthetic experiments and noise-perturbation sensitivity tests. The experiments use three families of synthetic targets: Gaussian-to-Gaussian, Symmetric Gaussian Ring, and Mixture of Gaussians.

This codebase runs those RFM experiments across curvatures (Euclidean / spherical / hyperbolic) using Hydra. The simplest entrypoint is the `general_fm` experiment in `riemannian-fm/configs/experiment/general_fm.yaml`.

## Results

The results reported in the paper can be viewed and reproduced in the `riemannian-fm/results/results.ipynb` notebook.

## Installation

Create the conda environment:

```bash
conda env create -f riemannian-fm/environment.yml
conda activate manifm
```

Install PyTorch (choose the command matching your CUDA / CPU setup). Example for CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Repository layout

| Path | What it contains |
| --- | --- |
| `riemannian-fm/train.py` | Main Hydra entrypoint for training (run with `python train.py experiment=...`). |
| `riemannian-fm/configs/train.yaml` | Base Hydra config (logging, output dirs, default optimizer/model settings). |
| `riemannian-fm/configs/experiment/general_fm.yaml` | Core synthetic experiment used in this repo (`data: general_fm`). |
| `riemannian-fm/configs/generate_mog_config.py` | CLI tool to generate MoG experiment YAMLs into `riemannian-fm/configs/experiment/`. |
| `riemannian-fm/manifm/` | Library code: datasets, manifolds, model components, Lightning module, metrics. |
| `riemannian-fm/results/results.ipynb` | Notebook used to view/reproduce paper results. |
| `riemannian-fm/results/runs/` | Example run artifacts used by the results/visualization utilities. |
| `riemannian-fm/outputs/` | Default Hydra output root for runs and multiruns. |

## Train (general_fm)

Run training from the RFM subproject:

```bash
cd riemannian-fm
python train.py experiment=general_fm
```

Common overrides:

```bash
# Disable Weights & Biases
python train.py experiment=general_fm use_wandb=False

# Switch manifold / curvature / dimension
python train.py experiment=general_fm general.manifold=euclidean general.dim=3
python train.py experiment=general_fm general.manifold=poincare  general.curvature=1.0 general.dim=3
python train.py experiment=general_fm general.manifold=sphere    general.curvature=1.0 general.dim=3

# Use different target distribution
python train.py experiment=general_fm general.x1_dist=gaussian-ring
python train.py experiment=general_fm general.x1_dist=checkerboard

# Enable tangent normalization of sampled distributions
python train.py experiment=general_fm general.normalize_tangent_distributions=True
```

*N.B.: Curvature for sphere is defined in ambient space*

Outputs:
- Hydra runs inside an auto-created output folder (see `hydra.run.dir` in `riemannian-fm/configs/train.yaml`).
- Checkpoints are saved under `checkpoints/` in the run directory.
- Metrics are written to `metrics.json` in the run directory.

## MoG config generator

Writing Mixture-of-Gaussians (MoG) experiments by hand in YAML can be tedious and error-prone. The generator script creates a complete Hydra experiment YAML for you.

Key concept:
- We define Gaussians in the tangent space at the origin, $T_0M$ (a flat Euclidean space).
- A specified `radius` is interpreted as a geodesic distance on the manifold.
- During training, the exponential map $\mathrm{Exp}_0(v)$ maps tangent vectors to points on the curved manifold.

### Generate a MoG experiment YAML

Run the generator from the repository root (it writes into `riemannian-fm/configs/experiment/`):

```bash
python riemannian-fm/configs/generate_mog_config.py \
	--filename my_mog.yaml \
	--manifold poincare \
	--curvature 1.0 \
	--dim 2 \
	--radii 1.5 1.5 \
	--angles 0 180 \
	--stds 0.1 0.1 \
	--weights 1.0 1.0
```

Notes:
- You must provide either:
	- Polar mode: `--radii ... --angles ...` (one radius+angle per Gaussian), or
	- Cartesian mode: `--cartesian_means '[[x1,y1,...],[x2,y2,...]]'`
- `--stds` can be isotropic (`0.1`) or anisotropic (`"0.5,0.1,0.1"`).
- Weights are normalized to sum to 1.0.

Optional per-component overrides (targets Gaussians by index: `G0`, `G1`, ...):

```bash
python riemannian-fm/configs/generate_mog_config.py \
	--filename my_mog_override.yaml \
	--manifold poincare \
	--dim 2 \
	--radii 1.5 1.5 \
	--angles 0 180 \
	--stds 0.1 0.1 \
	--weights 1.0 1.0 \
	--overrides '{"G0":{"weight":10.0,"std":[0.5,0.1]}}'
```

### Train using the generated config

Once `my_mog.yaml` exists under `riemannian-fm/configs/experiment/`, train it with:

```bash
cd riemannian-fm
python train.py experiment=my_mog
```

(Use the filename without the `.yaml`.)

## Acknowledgements

This project builds upon the following repositories:

- [facebookresearch/riemannian-fm](https://github.com/facebookresearch/riemannian-fm) — the original Riemannian Flow Matching implementation.
- [federicavaleau/Hyperbolic-Flow-Matching](https://github.com/federicavaleau/Hyperbolic-Flow-Matching) — hyperbolic flow matching extensions.
