# Experiments on ShapeNet

## Data

We use [ShapeNet][shapenet] data and their renderings, as provided by [R2N2][r2n2].

Run

```
datasets/shapenet/download_shapenet.sh
```

to download [R2N2][r2n2], and the train/val/test splits.
You also need the original ShapeNet Core v1 & binvox dataset, which require [registration][shapenet_login] before downloading.

## Preprocessing

```
python tools/preprocess_shapenet.py \
--shapenet_dir /path/to/ShapeNetCore.v1 \
--shapenet_binvox_dir /path/to/ShapeNetCore.v1.binvox \
--output_dir ./datasets/shapenet/ShapeNetV1processed \
--zip_output
```

The above command preprocesses the ShapeNet dataset to reduce the data loading time.
The preprocessed data will be saved in `./datasets/shapenet` and will be zipped.
The zipped output is useful when training in clusters.
