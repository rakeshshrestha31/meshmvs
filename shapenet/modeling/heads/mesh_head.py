# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv, SubdivideMeshes, vert_align
from torch.nn import functional as F

from shapenet.utils.coords import project_verts


class MeshRefinementHead(nn.Module):
    def __init__(self, cfg):
        super(MeshRefinementHead, self).__init__()

        # fmt: off
        input_channels  = cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS
        self.num_stages = cfg.MODEL.MESH_HEAD.NUM_STAGES
        hidden_dim      = cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM
        stage_depth     = cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS
        graph_conv_init = cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            vert_feat_dim = 0 if i == 0 else hidden_dim
            stage = MeshRefinementStage(
                input_channels, vert_feat_dim, hidden_dim, stage_depth, gconv_init=graph_conv_init
            )
            self.stages.append(stage)

    def forward(self, feats_extractor, meshes, P=None, subdivide=False):
        """
        Args:
          feats_extractor (function): return features given current mesh
          meshes (Meshes): Meshes class of N meshes
          P (list): list Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
          subdivide (bool): Flag whether to subdivice the mesh after refinement

        Returns:
          output_meshes (list of Meshes): A list with S Meshes, where S is the
                                          number of refinement stages
          features (list of dicts): features returned by feats_extractor
                                    for each mesh refinement stage
        """
        output_meshes = []
        output_feats = []
        vert_feats = None
        for i, stage in enumerate(self.stages):
            feats = feats_extractor(meshes)
            meshes, vert_feats = stage(feats["img_feats"], meshes, vert_feats, P)
            output_meshes.append(meshes)
            output_feats.append(feats)
            if subdivide and i < self.num_stages - 1:
                subdivide = SubdivideMeshes()
                meshes, vert_feats = subdivide(meshes, feats=vert_feats)
        return output_meshes, output_feats


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth, gconv_init="normal"):
        """
        Args:
          img_feat_dim (int): Dimension of features we will get from vert_align
          vert_feat_dim (int): Dimension of vert_feats we will receive from the
                               previous stage; can be 0
          hidden_dim (int): Output dimension for graph-conv layers
          stage_depth (int): Number of graph-conv layers to use
          gconv_init (int): Specifies weight initialization for graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.vert_offset = nn.Linear(hidden_dim + 3, 3)

        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.vert_offset.weight)
        nn.init.constant_(self.vert_offset.bias, 0)

    def forward(self, img_feats, meshes, vert_feats=None, Ps=None):
        """
        Args:
          img_feats (list): Features from the backbone
          meshes (Meshes): Initial meshes which will get refined
          vert_feats (tensor): Features from the previous refinement stage
          Ps (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
        """
        # Project verts if we are making predictions in world space
        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()

        vert_pos_padded, vert_pos_packed = [], []
        # find vertex pos in all views
        if Ps is not None:
            for P in Ps:
                if P is not None:
                    vert_pos_padded.append(
                        project_verts(meshes.verts_padded(), P)
                    )
                    vert_pos_packed.append(_padded_to_packed(
                        vert_pos_padded[-1], verts_padded_to_packed_idx
                    ))
        if not vert_pos_padded or not vert_pos_packed:
            vert_pos_padded.append(meshes.verts_padded())
            vert_pos_packed.append(meshes.verts_packed())
        # tensors of shape (B, V, N, 3)
        vert_pos_padded = torch.stack(vert_pos_padded, dim=1)
        # tensors of shape (V, N, 3)
        vert_pos_packed = torch.stack(vert_pos_packed, dim=0)
        vert_pos_packed_ref = vert_pos_packed[0]

        # for debug
        # save_meshes(vert_pos_packed, meshes)

        # flip y coordinate
        device, dtype = vert_pos_padded.device, vert_pos_padded.dtype
        factor = torch.tensor([1, -1, 1], device=device, dtype=dtype) \
                      .view(1, 1, 1, 3)
        vert_pos_padded = vert_pos_padded * factor
        # Get features from the image
        vert_align_feats = multi_view_vert_align(img_feats, vert_pos_padded)
        vert_align_feats = _padded_to_packed(
            vert_align_feats, verts_padded_to_packed_idx
        )
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        # Prepare features for first graph conv layer
        # Use the vertex coords from one view only
        first_layer_feats = [vert_align_feats, vert_pos_packed_ref]
        if vert_feats is not None:
            first_layer_feats.append(vert_feats)
        vert_feats = torch.cat(first_layer_feats, dim=1)

        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, meshes.edges_packed()))
            vert_feats = torch.cat(
                [vert_feats_nopos, vert_pos_packed_ref], dim=1
            )

        # Predict a new mesh by offsetting verts
        vert_offsets = torch.tanh(self.vert_offset(vert_feats))
        meshes_out = meshes.offset_verts(vert_offsets)

        return meshes_out, vert_feats_nopos


def multi_view_vert_align(img_feats, vert_pos_padded):
    """
    Extract multi-view features corresponding to mesh vertices
    from image features

    Args:
    - img_feats: list of tensors of shape (B, V, C, H, W)
    - vert_pos_padded: tensor of shape (B, V, N, 3)

    Returns:
    - list of tensors of shape (B, N, C). Length equal to length of img_feats
    """
    assert(torch.all(torch.tensor([
        feat.shape[1] == vert_pos_padded.shape[1] for feat in img_feats
    ])))
    vert_aligned_feats = []
    for view_idx, view_verts in enumerate(vert_pos_padded.unbind(1)):
        view_feats = [feat[:, view_idx] for feat in img_feats]
        vert_aligned_feats.append(vert_align(view_feats, view_verts))

    if len(vert_aligned_feats) == 1:
        # single view, just return the features
        return vert_aligned_feats[0]
    else:
        return extract_features_stats(vert_aligned_feats)


def extract_features_stats(features):
    """
    Extract feature statistics (max, mean, std) from a list of features

    Args:
    - vert_aligned_feats: list of tensors of shape (B, N, C)

    Returns:
    - tensor of shape (B, N, C*3)
    """
    joint_features = torch.stack(features, dim=-1)
    max_features = torch.max(joint_features, dim=-1)[0]
    mean_features = torch.mean(joint_features, dim=-1)
    var_features = torch.var(joint_features, dim=-1, unbiased=False)
    # calculating std using torch methods give NaN gradients
    # var will have different unit that mean/max, hence std desired
    std_features = torch.sqrt(var_features + 1e-8)
    return torch.cat((max_features, mean_features, std_features), dim=-1)


def _padded_to_packed(x, idx):
    """
    Convert features from padded to packed.

    Args:
      x: (N, V, D)
      idx: LongTensor of shape (VV,)

    Returns:
      feats_packed: (VV, D)
    """

    D = x.shape[-1]
    idx = idx.view(-1, 1).expand(-1, D)
    x_packed = x.view(-1, D).gather(0, idx)
    return x_packed


def save_meshes(vert_pos_packed, meshes):
    """
    Save meshes for debugging purpose

    Args:
    - vert_pos_packed: list of tensors of shape (V, N, 3)
    - meshes: Meshes in reference view i.e. view=0

    Returns:
    - None
    """
    from pytorch3d.io import save_obj
    import open3d as o3d
    import time
    timestamp = int(time.time() * 1000)

    for batch_idx, mesh in enumerate(meshes):
        for view_idx, vertices in enumerate(vert_pos_packed.unbind(0)):
            filename = "/tmp/proj_mesh_{}_{}_{}.obj" \
                            .format(timestamp, batch_idx, view_idx)
            save_obj(filename, vertices, mesh.faces_packed())
