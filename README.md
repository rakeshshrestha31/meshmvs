# MeshMVS

## Installation Requirements
- [Detectron2][d2]
- [PyTorch3D][py3d]

To install
```
pip install -e .
```

### ShapeNet Dataset preparation
See [INSTRUCTIONS_SHAPENET.md](INSTRUCTIONS_SHAPENET.md) for more instructions.

## Training
```
python tools/train_net_shapenet_depth.py \
  --num-gpus 5 --config-file configs/shapenet/voxmesh_R50_depth.yaml \
  SOLVER.NUM_EPOCHS 30 SOLVER.BATCH_SIZE 5 \
  OUTPUT_DIR output_depth_only
```

```
python tools/train_net_shapenet.py \
  --num-gpus 5 --config-file configs/shapenet/voxmesh_R50_depth.yaml \
  MODEL.MVSNET.CHECKPOINT output_depth_only/checkpoint_with_model.pt \
  MODEL.MVSNET.FREEZE True \
  MODEL.FEATURE_FUSION_METHOD multihead_attention \
  MODEL.MULTIHEAD_ATTENTION.FEATURE_DIMS 480 MODEL.MULTIHEAD_ATTENTION.NUM_HEADS 5 \
  SOLVER.BATCH_SIZE 5 SOLVER.BATCH_SIZE_EVAL 1 \
  MODEL.MVSNET.RENDERED_DEPTH_WEIGHT 0.001 MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT 0.0 \
  MODEL.MVSNET.RENDERED_VS_GT_DEPTH_WEIGHT 0.00 \
  MODEL.CONTRASTIVE_DEPTH_TYPE input_concat \
  MODEL.VOXEL_HEAD.RGB_FEATURES_INPUT True \
  MODEL.VOXEL_HEAD.DEPTH_FEATURES_INPUT False \
  MODEL.VOXEL_HEAD.RGB_BACKBONE resnet50 \
  MODEL.MESH_HEAD.RGB_FEATURES_INPUT True \
  MODEL.MESH_HEAD.DEPTH_FEATURES_INPUT True \
  OUTPUT_DIR output
```

## Evaluation
```
python tools/train_net_shapenet.py \
  --eval-only \
  --num-gpus 5 --config-file configs/shapenet/voxmesh_R50_depth.yaml \
  MODEL.MVSNET.CHECKPOINT output_depth_only/checkpoint_with_model.pt \
  MODEL.MVSNET.FREEZE True \
  MODEL.FEATURE_FUSION_METHOD multihead_attention \
  MODEL.MULTIHEAD_ATTENTION.FEATURE_DIMS 480 MODEL.MULTIHEAD_ATTENTION.NUM_HEADS 5 \
  SOLVER.BATCH_SIZE 5 SOLVER.BATCH_SIZE_EVAL 1 \
  MODEL.MVSNET.RENDERED_DEPTH_WEIGHT 0.001 MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT 0.0 \
  MODEL.MVSNET.RENDERED_VS_GT_DEPTH_WEIGHT 0.00 \
  MODEL.CONTRASTIVE_DEPTH_TYPE input_concat \
  MODEL.VOXEL_HEAD.RGB_FEATURES_INPUT True \
  MODEL.VOXEL_HEAD.DEPTH_FEATURES_INPUT False \
  MODEL.VOXEL_HEAD.RGB_BACKBONE resnet50 \
  MODEL.MESH_HEAD.RGB_FEATURES_INPUT True \
  MODEL.MESH_HEAD.DEPTH_FEATURES_INPUT True \
  OUTPUT_DIR output MODEL.CHECKPOINT output/checkpoint_with_model.pt

```

Install [Pixel2Mesh++](https://github.com/walsvid/Pixel2MeshPlusPlus) for f-score
```
python f_score.py --gpu_id 0 --save_path ~/projects/meshrcnn/output/predictions/ --name eval --epochs 0

```

