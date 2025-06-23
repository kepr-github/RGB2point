<p align="center">
    <img src="https://www.jaejoonglee.com/images/rgb2point.png" alt="Overview">
</p>

**RGB2Point** is officially accepted to WACV 2025. It takes a single unposed RGB image to generate 3D Point Cloud. Check more details from the [paper](https://arxiv.org/pdf/2407.14979).
## Codes
**RGB2Point** is tested on Ubuntu 22 and Windows 11. `Python 3.9+` and `Pytorch 2.0+` is required.

## Dependencies
Assuming `Pytorch 2.0+` with `CUDA` is installed, run:
```
pip install timm
pip install accelerate
pip install wandb
pip install scikit-learn
pip install gsplat
```
If `gsplat` needs to be compiled from source, make sure the GLM headers
are available. On Debian-based systems install `libglm-dev`:

```
apt-get update && apt-get install -y libglm-dev
```

## Docker
You can also run **RGB2Point** using Docker Compose. The provided compose file
mounts the repository into the container so that changes on the host are
immediately available.

Build the image and start an interactive session:

```
docker compose run --build --rm app
```

This container includes all Python dependencies required to train and infer
models.

## Training
By default the training script expects the dataset to be located under
`data/` with the two folders `ShapeNet_pointclouds` and
`ShapeNetRendering` inside. Launch training with:

```bash
python train.py
```

If your data resides elsewhere or you want to train on a different set of
categories, provide the dataset root and a list of category names. For example,
to train on a custom category named `soybean` placed under
`data/ShapeNetRendering/soybean` and `data/ShapeNet_pointclouds/soybean` run:

```bash
python train.py --root /path/to/data --categories soybean
```
To use multiple categories list them all after `--categories`.

## Training Data
Please download 1)  [Point cloud data zip file](https://drive.google.com/file/d/1R7TXnBvVir8OCXPE5f2kck6Enl0gdMUQ/view?usp=sharing), 2) [Rendered Images](https://drive.google.com/file/d/1t_rlV1BwitvICap_2ubd5oqL_6Yq-Drn/view?usp=sharing), and 3) [Train/test filenames](https://drive.google.com/drive/folders/1jBPd1YBJwzgVpolT-yA0g8XxYJmb2_s-?usp=sharing).

Place the extracted folders `ShapeNet_pointclouds` and `ShapeNetRendering` under
the directory specified by `--root` (``data/`` by default). The optional files
`shapenet_train.txt` and `shapenet_test.txt` should also be located in this
root directory. Each line of these files lists a model using the format
``<category>/<model_name>``. When present, the loader will only use the models
listed in the corresponding file for the train or test split. If the files are
omitted, all available models are used.

## Pretrained Model
Download the model trained on Chair, Airplane and Car from ShapeNet.
```
https://drive.google.com/file/d/1Z5luy_833YV6NGiKjGhfsfEUyaQkgua1/view?usp=sharing
```

## Inference
```
python inference.py
```
Change `image_path` and `save_path` in `inference.py` accrodingly.


## Preparing gs data
If you have a directory `gs/` that contains `.ply` files grouped by
category, or even a single `.ply` file, you can convert it into the
ShapeNet-style format used in this repository. Run:

```
python prepare_gs.py ./gs data
```

This generates `ShapeNet_pointclouds` and `ShapeNetRendering` folders
inside `data/`. For each model, two point clouds (`pointcloud_1024.npy`
and `pointcloud_2048.npy`) are created together with 24 randomly
rendered views using `gsplat` saved under a `rendering` directory. You can also pass a
single `.ply` file instead of a directory; the parent folder name will
be used as its category.

## Rendering gs data
If you want to quickly visualize a Scaniverse-style Gaussian `.ply` file you can
use the `gsplat` renderer. First install the dependency:

```
pip install gsplat
```

Render the file with:

```
python render_gs.py your_scaniverse.ply output.png
```

This will generate an image of the Gaussian model using the default camera
parameters.

## Converting npy point clouds to PLY
Point clouds produced by `prepare_gs.py` are stored as NumPy arrays. To view
them in CloudCompare or other point cloud viewers, convert the file to PLY:

```
python convert_npy_to_ply.py data/ShapeNet_pointclouds/gs/エンレイ2/pointcloud_2048.npy output.ply
```

Replace the paths with your own `.npy` file and desired output location.



## update
We have added support for the BlendedMVS Dataset with a custom DataLoader located in the notebooks folder. This integration enhances multi-view stereo capability for 3D reconstruction.

## Reference
If you find this paper and code useful in your research, please consider citing:
```bibtex
@InProceedings{Lee_2025_WACV,
    author    = {Lee, Jae Joong and Benes, Bedrich},
    title     = {RGB2Point: 3D Point Cloud Generation from Single RGB Images},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2952-2962}
}
```
