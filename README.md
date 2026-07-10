# SeaIceGS

SeaIceGS is a research codebase for polar region 3D reconstruction from multi-view imagery. It adapts Gaussian Splatting-based surface reconstruction to polar  scenes, where weak texture, high reflectance, and fragmented ice geometry make dense reconstruction difficult.

The project takes image sequences and camera poses as input, trains a Gaussian scene representation, renders RGB/depth/normal views, and exports a mesh through TSDF fusion. The generated DOM/DSM products can be used by downstream sea-ice retrieval workflows such as SIC and freeboard estimation.

## Project Layout

```text
SeaIceGS/
  arguments/              # command-line parameter groups
  gaussian_renderer/      # Gaussian rasterization and rendering interface
  scene/                  # camera, dataset, and Gaussian model definitions
  utils/                  # geometry, image, loss, and system utilities
  submodules/             # CUDA extensions used by the renderer
  train.py                # training entry point
  render.py               # rendering and mesh extraction entry point
  DigitalProduct.py       # DOM and DSM generation from exported mesh
  metrics.py              # image quality metrics
  TSDF.py                 # TSDF fusion utility
  requirements.txt
```

## Installation

Clone the repository:

```bash
git clone git@github.com:QuanJianji/SeaIceGS.git
cd SeaIceGS
```

Create and activate a conda environment:

```bash
conda create -n SeaIceGS python=3.8 -y
conda activate SeaIceGS
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install Python dependencies and CUDA extensions:

```bash
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

Download the VGG16 checkpoint used by LPIPS:

```bash
mkdir -p vgg_checkpoint
wget https://download.pytorch.org/models/vgg16-397923af.pth -O vgg_checkpoint/vgg16-397923af.pth
```

## Data Preparation

SeaIceGS expects a reconstructed image dataset with camera parameters. A typical custom dataset can be arranged as:

```text
data/
  scene_name/
    images/
      image_0001.jpg
      image_0002.jpg
      ...
    murre_out/
      depth_bw/
      depth_colored/
      depth_npy/
        image_0001_pred.npy
        image_0002_pred.npy
        ...
    sparse/
      0/
        cameras.bin
        images.bin
        points3D.bin
```
 For murre_out folder,
 For polar sea-ice imagery, DOM and DSM products can also be generated with SeaIceGS outputs or with photogrammetry software such as Metashape Professional.

## Training

Run training with:

```bash
python train.py \
  -s /path/to/data/scene_name \
  -m /path/to/output/scene_name \
  --max_abs_split_points 0 \
  --opacity_cull_threshold 0.05
```

Main parameters are defined in `arguments/__init__.py`.

## Rendering and Mesh Extraction

To extract a mesh with TSDF fusion:

```bash
python render.py \
  -m /path/to/output/scene_name \
  --max_depth 500.0 \
  --voxel_size 0.01 \
```

## Digital Product Generation

After mesh extraction, the 3DGS output directory should contain a `mesh/` folder. Generate DOM and DSM GeoTIFF products with:

```bash
python DigitalProduct.py \
  --input /path/to/output/scene_name \
  --resolution 0.4
```

The script searches `mesh/` for `tsdf_fusion_scaled.ply`, `tsdf_fusion_post.ply`, or `tsdf_fusion.ply` in order, and writes:

```text
/path/to/output/scene_name/digital_products/DOM.tif
/path/to/output/scene_name/digital_products/DSM.tif
```

If multiple mesh files exist and a specific one is needed, specify it with `--mesh-name`.

## Sea Ice Parameters retrieval

For the retrieval of SIC and relative SIF, please follow the lead of [IceRetrieval](https://github.com/QuanJianji/IceRetrieval)

## Notes for Sea-Ice Scenes

- Use high-overlap imagery when possible; sea ice often contains low-texture and repeated patterns.
- Adjust `--max_depth` and `--voxel_size` according to scene scale.
- For noisy or incomplete viewpoints, enable `--use_depth_filter` during mesh extraction.
- Reduce `--opacity_cull_threshold` carefully when thin ice fragments or small floes need to be preserved.

## Acknowledgements

SeaIceGS is built on top of prior work including:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [PGSR](https://github.com/zju3dv/PGSR)
- [Murre](https://github.com/zju3dv/Murre)
- [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything)
- COLMAP, related evaluation/preprocessing tools and ShetlandsUAVmetry: Deception Island

Please follow the original licenses and citation requirements of these projects.

## License

This repository contains code derived from third-party research projects. Use is governed by `LICENSE.md` and the licenses of included third-party components. Copyright and attribution notices must be retained.
