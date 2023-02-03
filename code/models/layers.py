import torch
from torch.nn import Linear, ReLU, LayerNorm, Sequential, Module, Identity
from torch.nn import functional as F
import torch_geometric
from utils.sparse_utils import SparseMat
from utils.pos_enc_utils import get_embedder
from utils import sparse_utils


def get_linear_layers(feats, init_activation=False, final_activation=False, norm=True):
    assert len(feats) >= 2

    layers = []

    # Initial activation
    if init_activation:
        if norm:
            # layers.append(BatchNorm1d(feats[-1], track_running_stats=False))
            layers.append(LayerNorm(feats[0]))

        layers.append(ReLU(inplace=True))

    # Add layers
    for i in range(len(feats) - 2):
        layers.append(Linear(feats[i], feats[i + 1]))

        if norm:
            # layers.append(BatchNorm1d(feats[i + 1], track_running_stats=False))
            layers.append(LayerNorm(feats[i + 1]))

        layers.append(ReLU(inplace=True))

    # Add final layer
    layers.append(Linear(feats[-2], feats[-1]))

    # Final activation
    if final_activation:
        if norm:
            # layers.append(BatchNorm1d(feats[-1], track_running_stats=False))
            layers.append(LayerNorm(feats[-1]))

        layers.append(ReLU(inplace=True))

    return Sequential(*layers)


class Parameter3DPts(torch.nn.Module):
    def __init__(self, n_pts):
        super().__init__()

        # Init points randomly
        pts_3d = torch.normal(mean=0, std=0.1, size=(3, n_pts), requires_grad=True)

        self.pts_3d = torch.nn.Parameter(pts_3d)

    def forward(self):
        return self.pts_3d


# class SetOfSetLayer(Module):
#     def __init__(self, d_in, d_out):
#         super(SetOfSetLayer, self).__init__()
#         # n is the number of points and m is the number of cameras
#         self.lin_proj = Linear(d_in, d_out)
#         self.lin_scenepoint = Linear(d_in, d_out)
#         self.lin_view = Linear(d_in, d_out)
#         self.lin_global = Linear(d_in, d_out)

#     def forward(self, x):
#         # x is [m,n,d] sparse matrix
#         proj_features = self.lin_proj(x.values)  # [nnz,d_in] -> [nnz,d_out]

#         mean_colwise = x.mean(dim=0) # [m,n,d_in] -> [n,d_in]
#         scenepoint_features = self.lin_scenepoint(mean_colwise)  # [n,d_in] -> [n,d_out]

#         mean_rowwise = x.mean(dim=1) # [m,n,d_in] -> [m,d_in]
#         view_features = self.lin_view(mean_rowwise)  # [m,d_in] -> [m,d_out]

#         global_features = self.lin_global(x.values.mean(dim=0, keepdim=True))  # [1,d_in] -> [1,d_out]

#         new_features = (proj_features + scenepoint_features[x.indices[1], :] + view_features[x.indices[0], :] + global_features) / 4  # [nnz,d_out]
#         new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

#         return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class SetOfSetLayer(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetLayer, self).__init__()
        self.global_feature_update = SetOfSetGlobalFeatureUpdate(d_in, d_out)
        self.projection_feature_update = SetOfSetProjectionFeatureUpdate(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        scenepoint_features, view_features, global_features = self.global_feature_update(x)
        x = self.projection_feature_update(scenepoint_features, view_features, global_features, x)
        return x


class SetOfSetGlobalFeatureUpdate(Module):
    def __init__(self, d_in, d_out, output_global=True):
        super(SetOfSetGlobalFeatureUpdate, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_scenepoint = Linear(d_in, d_out)
        self.lin_view = Linear(d_in, d_out)
        self.output_global = output_global
        if output_global:
            self.lin_global = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix

        # Within this method, let's use a conventional pytorch sparse tensor, rather than the custom SparseMat.
        x = x.to_torch_hybrid_sparse_coo()

        mean_colwise = sparse_utils.sparse_mean(x, dim=0).to_dense() # [m,n,d_in] -> [n,d_in]
        scenepoint_features = self.lin_scenepoint(mean_colwise)  # [n,d_in] -> [n,d_out]

        mean_rowwise = sparse_utils.sparse_mean(x, dim=1).to_dense() # [m,n,d_in] -> [m,d_in]
        view_features = self.lin_view(mean_rowwise)  # [m,d_in] -> [m,d_out]

        if not self.output_global:
            return scenepoint_features, view_features
        else:
            global_features = self.lin_global(sparse_utils.sparse_mean(x, dim=(0, 1))[None, :].to_dense())  # [1,d_in] -> [1,d_out]
            return scenepoint_features, view_features, global_features


class SetOfSetProjectionFeatureUpdate(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetProjectionFeatureUpdate, self).__init__()
        self.lin_proj = Linear(d_in, d_out)

    def forward(self, scenepoint_features, view_features, global_features, x):
        # x is [m,n,d] sparse matrix

        # Within this method, let's use a conventional pytorch sparse tensor, rather than the custom SparseMat.
        cam_per_pts, pts_per_cam = x.cam_per_pts, x.pts_per_cam
        x = x.to_torch_hybrid_sparse_coo()

        proj_features = self.lin_proj(x.values())  # [nnz,d_in] -> [nnz,d_out]

        new_features = (proj_features + scenepoint_features[x.indices()[1], :] + view_features[x.indices()[0], :] + global_features) / 4  # [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        # NOTE: Importantly, we do not mix values / indices of the new tensor with those of the input SparseMat, as the order may have changed due to .coalesce().
        return SparseMat(new_features, x.indices(), cam_per_pts, pts_per_cam, new_shape)


class GraphAttnSfMLayer(Module):
    def __init__(
        self,
        n_feat_proj_in,
        n_feat_proj_out,
        n_feat_scenepoint_hidden,
        n_feat_view_hidden,
        n_feat_global_hidden,
        n_feat_proj2scenepoint_agg = None,
        n_feat_proj2view_agg = None,
        n_feat_scenepoint2global_agg = None,
        n_feat_view2global_agg = None,
        use_norm_proj_update = True,
        add_residual_skipconn_proj_update = True,
        n_feat_skipconn_init_projfeat_in = None,
        n_heads = 1,
        stateful = True,
        global2view_and_global2scenepoint_enabled = True,
        n_hidden_layers_scenepoint_update = 0,
        n_hidden_layers_view_update = 0,
        n_hidden_layers_global_update = 0,
        n_hidden_layers_proj_update = 0,
    ):
        super(GraphAttnSfMLayer, self).__init__()

        self.use_norm_proj_update = use_norm_proj_update
        self.add_residual_skipconn_proj_update = add_residual_skipconn_proj_update

        if n_feat_skipconn_init_projfeat_in is not None:
            self.add_skipconn_from_init_projfeat = True
            self.n_feat_skipconn_init_projfeat_in = n_feat_skipconn_init_projfeat_in
        else:
            self.add_skipconn_from_init_projfeat = False
            self.n_feat_skipconn_init_projfeat_in = 0

        if self.use_norm_proj_update:
            self.prev_projfeat_norm_layer = LayerNorm(n_feat_proj_in)

        self.global_feature_update = GraphAttnSfMGlobalFeatureUpdate(
            n_feat_proj_in,
            n_feat_scenepoint_hidden,
            n_feat_view_hidden,
            n_feat_proj2scenepoint_agg = n_feat_proj2scenepoint_agg,
            n_feat_proj2view_agg = n_feat_proj2view_agg,
            n_feat_global_out = n_feat_global_hidden,
            n_feat_scenepoint2global_agg = n_feat_scenepoint2global_agg,
            n_feat_view2global_agg = n_feat_view2global_agg,
            output_global = True,
            n_heads = n_heads,
            stateful = stateful,
            global2view_and_global2scenepoint_enabled = global2view_and_global2scenepoint_enabled,
            n_hidden_layers_scenepoint_update = n_hidden_layers_scenepoint_update,
            n_hidden_layers_view_update = n_hidden_layers_view_update,
            n_hidden_layers_global_update = n_hidden_layers_global_update,
        )
        self.projection_feature_update = GraphAttnSfMProjectionFeatureUpdate(
            n_feat_proj_in + self.n_feat_skipconn_init_projfeat_in,
            n_feat_scenepoint_hidden,
            n_feat_view_hidden,
            n_feat_global_hidden,
            n_feat_proj_out,
            n_hidden_layers_proj_update = n_hidden_layers_proj_update,
            normalize_global_features = True,
        )
        if self.add_residual_skipconn_proj_update:
            if n_feat_proj_in == n_feat_proj_out:
                self.skip_projection = None
            else:
                if self.use_norm_proj_update:
                    self.residual_skipconn_proj_norm_layer = LayerNorm(n_feat_proj_in)
                self.skip_projection = ProjLayer(n_feat_proj_in, n_feat_proj_out)

    def forward(
        self,
        prev_projection_features,
        graph_structure,
        prev_scenepoint_features = None,
        prev_view_features = None,
        prev_global_features = None,
        skipconn_init_projfeat = None,
    ):
        prev_projection_features_raw = prev_projection_features
        if self.use_norm_proj_update:
            prev_projection_features = normalize_projection_features(prev_projection_features, norm_layer=self.prev_projfeat_norm_layer)
        prev_projection_features = relu_on_projection_features(prev_projection_features)

        # prev_projection_features is [m,n,d] sparse matrix
        scenepoint_features, view_features, global_features = self.global_feature_update(
            prev_projection_features,
            graph_structure,
            prev_scenepoint_features = prev_scenepoint_features,
            prev_view_features = prev_view_features,
            prev_global_features = prev_global_features,
        )
        projection_features = prev_projection_features
        if self.add_skipconn_from_init_projfeat:
            assert skipconn_init_projfeat is not None
            assert skipconn_init_projfeat.values.shape[1] == self.n_feat_skipconn_init_projfeat_in
            projection_features = sparse_utils.sparsemat_feature_cat([
                projection_features,
                skipconn_init_projfeat,
            ])
        projection_features = self.projection_feature_update(scenepoint_features, view_features, global_features, projection_features)

        if self.add_residual_skipconn_proj_update:
            x_skip = prev_projection_features_raw
            if self.skip_projection is not None:
                if self.use_norm_proj_update:
                    x_skip = normalize_projection_features(x_skip, norm_layer=self.residual_skipconn_proj_norm_layer)
                    x_skip = relu_on_projection_features(x_skip)
                x_skip = self.skip_projection(x_skip)
            projection_features = x_skip + projection_features

        return projection_features, scenepoint_features, view_features, global_features


class Proj2View(Module):
    """
    Resection layer:
    Projection feature -> view feature aggregation.
    """
    def __init__(
        self,
        n_feat_proj_in,
        n_feat_view_out,
        n_heads,
        stateful = True,
        use_norm_pre_mlp = True,
        n_feat_proj2view_agg = None,
        n_hidden_layers_view_update = 0,
    ):
        super(Proj2View, self).__init__()
        self.n_feat_proj_in = n_feat_proj_in
        self.n_feat_view_out = n_feat_view_out
        self.stateful = stateful
        self.use_norm_pre_mlp = use_norm_pre_mlp

        if n_feat_proj2view_agg is None:
            n_feat_proj2view_agg = n_feat_proj_in # Default
            if n_feat_proj2view_agg % n_heads > 0:
                n_feat_proj2view_agg += n_heads - (n_feat_proj2view_agg % n_heads)

        assert n_feat_proj2view_agg % n_heads == 0
        self.n_feat_proj2view_agg = n_feat_proj2view_agg

        if self.stateful:
            self.norm_and_proj_view2proj = []
            use_norm_pre_proj_view2proj = True
            if use_norm_pre_proj_view2proj:
                self.norm_and_proj_view2proj.append(LayerNorm(self.n_feat_view_out))
                self.norm_and_proj_view2proj.append(ReLU(inplace=True))
            if self.n_feat_proj_in != self.n_feat_view_out:
                self.norm_and_proj_view2proj.append(Linear(self.n_feat_view_out, self.n_feat_proj_in))
            self.norm_and_proj_view2proj = Sequential(*self.norm_and_proj_view2proj)
        self.graph_conv = torch_geometric.nn.GATv2Conv(
            n_feat_proj_in,
            n_feat_proj2view_agg // n_heads,
            heads = n_heads,
            add_self_loops = False,
        )
        if n_feat_proj2view_agg != n_feat_view_out:
            self.proj_proj2view = Linear(n_feat_proj2view_agg, n_feat_view_out)
        if self.use_norm_pre_mlp:
            self.norm_pre_mlp = LayerNorm(n_feat_view_out)
        self.mlp = get_linear_layers(
            (2 + n_hidden_layers_view_update) * [n_feat_view_out],
            init_activation = False,
            final_activation = False,
            norm = False,
        )

    def forward(
        self,
        proj_features, # projection features, normalized, to be fed as inputs to the "cross-attention" residual mappings for view features and scenepoint features, respectively.
        graph_wrapper,
        prev_view_features = None,
    ):
        assert self.stateful == (prev_view_features is not None)

        x = self.graph_conv(
            graph_wrapper.generate_node_features(
                proj_features,
                x_agg = None if prev_view_features is None else self.norm_and_proj_view2proj(prev_view_features),
            ),
            graph_wrapper.edge_index,
        )
        x = graph_wrapper.extract_target_node_features(x)
        assert x.ndim == 3
        x = x.squeeze(1) # (m, 1, n_feat_proj2view_agg) -> (m, n_feat_proj2view_agg)
        assert x.ndim == 2

        if self.n_feat_proj2view_agg != self.n_feat_view_out:
            x = self.proj_proj2view(x)

        if prev_view_features is not None:
            x_skip = prev_view_features
            assert x.shape == x_skip.shape # NOTE: If False, we need to implement projection of skip connection to expected dimensionality.
            x = x_skip + x

        # Apply MLP
        x_skip = x
        if self.use_norm_pre_mlp:
            x = self.norm_pre_mlp(x)
            x = F.relu(x)
        x = self.mlp(x)
        assert x.ndim == 2 and x.shape[1] == self.n_feat_view_out
        assert x_skip.ndim == 2 and x_skip.shape[1] == self.n_feat_view_out # NOTE: If False, we need to implement projection of skip connection to expected dimensionality.
        x = x_skip + x

        view_features = x

        return view_features

class Proj2ScenePoint(Module):
    """
    Intersection layer:
    Projection feature -> scenepoint feature aggregation.
    """
    def __init__(
        self,
        n_feat_proj_in,
        n_feat_scenepoint_out,
        n_heads,
        stateful = True,
        use_norm_pre_mlp = True,
        n_feat_proj2scenepoint_agg = None,
        n_hidden_layers_scenepoint_update = 0,
    ):
        super(Proj2ScenePoint, self).__init__()
        self.n_feat_proj_in = n_feat_proj_in
        self.n_feat_scenepoint_out = n_feat_scenepoint_out
        self.stateful = stateful
        self.use_norm_pre_mlp = use_norm_pre_mlp

        if n_feat_proj2scenepoint_agg is None:
            n_feat_proj2scenepoint_agg = n_feat_proj_in # Default
            if n_feat_proj2scenepoint_agg % n_heads > 0:
                n_feat_proj2scenepoint_agg += n_heads - (n_feat_proj2scenepoint_agg % n_heads)

        assert n_feat_proj2scenepoint_agg % n_heads == 0
        self.n_feat_proj2scenepoint_agg = n_feat_proj2scenepoint_agg

        if self.stateful:
            self.norm_and_proj_scenepoint2proj = []
            use_norm_pre_proj_scenepoint2proj = True
            if use_norm_pre_proj_scenepoint2proj:
                self.norm_and_proj_scenepoint2proj.append(LayerNorm(self.n_feat_scenepoint_out))
                self.norm_and_proj_scenepoint2proj.append(ReLU(inplace=True))
            if self.n_feat_proj_in != self.n_feat_scenepoint_out:
                self.norm_and_proj_scenepoint2proj.append(Linear(self.n_feat_scenepoint_out, self.n_feat_proj_in))
            self.norm_and_proj_scenepoint2proj = Sequential(*self.norm_and_proj_scenepoint2proj)
        self.graph_conv = torch_geometric.nn.GATv2Conv(
            n_feat_proj_in,
            n_feat_proj2scenepoint_agg // n_heads,
            heads = n_heads,
            add_self_loops = False,
        )
        if n_feat_proj2scenepoint_agg != n_feat_scenepoint_out:
            self.proj_proj2scenepoint = Linear(n_feat_proj2scenepoint_agg, n_feat_scenepoint_out)
        if self.use_norm_pre_mlp:
            self.norm_pre_mlp = LayerNorm(n_feat_scenepoint_out)
        self.mlp = get_linear_layers(
            (2 + n_hidden_layers_scenepoint_update) * [n_feat_scenepoint_out],
            init_activation = False,
            final_activation = False,
            norm = False,
        )

    def forward(
        self,
        proj_features, # projection features, normalized, to be fed as inputs to the "cross-attention" residual mappings for view features and scenepoint features, respectively.
        graph_wrapper,
        prev_scenepoint_features = None,
    ):
        assert self.stateful == (prev_scenepoint_features is not None)

        x = self.graph_conv(
            graph_wrapper.generate_node_features(
                proj_features,
                x_agg = None if prev_scenepoint_features is None else self.norm_and_proj_scenepoint2proj(prev_scenepoint_features),
            ),
            graph_wrapper.edge_index,
        )
        x = graph_wrapper.extract_target_node_features(x)
        assert x.ndim == 3
        x = x.squeeze(0) # (1, n, n_feat_proj2scenepoint_agg) -> (n, n_feat_proj2scenepoint_agg)
        assert x.ndim == 2

        if self.n_feat_proj2scenepoint_agg != self.n_feat_scenepoint_out:
            x = self.proj_proj2scenepoint(x)

        if prev_scenepoint_features is not None:
            x_skip = prev_scenepoint_features
            assert x.shape == x_skip.shape # NOTE: If False, we need to implement projection of skip connection to expected dimensionality.
            x = x_skip + x

        # Apply MLP
        x_skip = x
        if self.use_norm_pre_mlp:
            x = self.norm_pre_mlp(x)
            x = F.relu(x)
        x = self.mlp(x)
        assert x.ndim == 2 and x.shape[1] == self.n_feat_scenepoint_out
        assert x_skip.ndim == 2 and x_skip.shape[1] == self.n_feat_scenepoint_out # NOTE: If False, we need to implement projection of skip connection to expected dimensionality.
        x = x_skip + x

        scene_features = x

        return scene_features

class ViewAndScenePoint2Global(Module):
    """
    Global aggregation layer from view features & scenepoint features.
    """
    def __init__(
        self,
        n_feat_scenepoint_in,
        n_feat_view_in,
        n_feat_global_out,
        n_heads,
        stateful = True,
        use_norm_pre_mlp = True,
        n_feat_scenepoint2global_agg = None,
        n_feat_view2global_agg = None,
        n_hidden_layers_global_update = 0,
    ):
        super(ViewAndScenePoint2Global, self).__init__()
        self.n_feat_scenepoint_in = n_feat_scenepoint_in
        self.n_feat_view_in = n_feat_view_in
        self.n_feat_global_out = n_feat_global_out
        self.stateful = stateful
        self.use_norm_pre_mlp = use_norm_pre_mlp
        self.n_feat_scenepoint2global_agg = n_feat_scenepoint2global_agg
        self.n_feat_view2global_agg = n_feat_view2global_agg

        if self.n_feat_scenepoint2global_agg is None:
            self.n_feat_scenepoint2global_agg = self.n_feat_scenepoint_in # Default
            if self.n_feat_scenepoint2global_agg % n_heads > 0:
                self.n_feat_scenepoint2global_agg += n_heads - (self.n_feat_scenepoint2global_agg % n_heads)
        if self.n_feat_view2global_agg is None:
            self.n_feat_view2global_agg = self.n_feat_view_in # Default
            if self.n_feat_view2global_agg % n_heads > 0:
                self.n_feat_view2global_agg += n_heads - (self.n_feat_view2global_agg % n_heads)

        assert self.n_feat_scenepoint2global_agg % n_heads == 0
        assert self.n_feat_view2global_agg % n_heads == 0

        if self.stateful:
            self.norm_and_proj_global2view = []
            use_norm_pre_proj_global2view = True
            if use_norm_pre_proj_global2view:
                self.norm_and_proj_global2view.append(LayerNorm(self.n_feat_global_out))
                self.norm_and_proj_global2view.append(ReLU(inplace=True))
            if self.n_feat_view_in != self.n_feat_global_out:
                self.norm_and_proj_global2view.append(Linear(self.n_feat_global_out, self.n_feat_view_in))
            self.norm_and_proj_global2view = Sequential(*self.norm_and_proj_global2view)
        self.graph_conv_view2global = torch_geometric.nn.GATv2Conv(
            self.n_feat_view_in,
            self.n_feat_view2global_agg // n_heads,
            heads = n_heads,
            add_self_loops = False,
        )
        if self.stateful:
            self.norm_and_proj_global2scenepoint = []
            use_norm_pre_proj_global2scenepoint = True
            if use_norm_pre_proj_global2scenepoint:
                self.norm_and_proj_global2scenepoint.append(LayerNorm(self.n_feat_global_out))
                self.norm_and_proj_global2scenepoint.append(ReLU(inplace=True))
            if self.n_feat_scenepoint_in != self.n_feat_global_out:
                self.norm_and_proj_global2scenepoint.append(Linear(self.n_feat_global_out, self.n_feat_scenepoint_in))
            self.norm_and_proj_global2scenepoint = Sequential(*self.norm_and_proj_global2scenepoint)
        self.graph_conv_scenepoint2global = torch_geometric.nn.GATv2Conv(
            self.n_feat_scenepoint_in,
            self.n_feat_scenepoint2global_agg // n_heads,
            heads = n_heads,
            add_self_loops = False,
        )
        if (self.n_feat_view2global_agg + self.n_feat_scenepoint2global_agg) != n_feat_global_out:
            self.proj_view_and_scenepoint2global = Linear(self.n_feat_view2global_agg + self.n_feat_scenepoint2global_agg, n_feat_global_out)
        if self.use_norm_pre_mlp:
            self.norm_pre_mlp = LayerNorm(n_feat_global_out)
        self.mlp = get_linear_layers(
            (2 + n_hidden_layers_global_update) * [n_feat_global_out],
            init_activation = False,
            final_activation = False,
            norm = False,
        )

    def forward(self, view_features, scenepoint_features, graph_wrapper_view2global, graph_wrapper_scenepoint2global, prev_global_features=None):
        assert self.stateful == (prev_global_features is not None)

        view_features = view_features[:, None, :] # (m, 1, n_feat_view_out)
        scenepoint_features = scenepoint_features[None, :, :] # (1, n, n_feat_scenepoint_out)

        # Aggregate view features
        view_features = torch.sparse_coo_tensor(
            graph_wrapper_view2global.valid_indices,
            view_features[graph_wrapper_view2global.valid_indices[0], graph_wrapper_view2global.valid_indices[1], :],
            size = view_features.shape,
        ).coalesce()
        view2global_features = self.graph_conv_view2global(
            graph_wrapper_view2global.generate_node_features(
                view_features,
                x_agg = None if prev_global_features is None else self.norm_and_proj_global2view(prev_global_features),
            ),
            graph_wrapper_view2global.edge_index,
        )
        view2global_features = graph_wrapper_view2global.extract_target_node_features(view2global_features)
        assert view2global_features.shape == (1, 1, self.n_feat_view2global_agg)

        # Aggregate scenepoint features
        scenepoint_features = torch.sparse_coo_tensor(
            graph_wrapper_scenepoint2global.valid_indices,
            scenepoint_features[graph_wrapper_scenepoint2global.valid_indices[0], graph_wrapper_scenepoint2global.valid_indices[1], :],
            size = scenepoint_features.shape,
        ).coalesce()
        scenepoint2global_features = self.graph_conv_scenepoint2global(
            graph_wrapper_scenepoint2global.generate_node_features(
                scenepoint_features,
                x_agg = None if prev_global_features is None else self.norm_and_proj_global2scenepoint(prev_global_features),
            ),
            graph_wrapper_scenepoint2global.edge_index,
        )
        scenepoint2global_features = graph_wrapper_scenepoint2global.extract_target_node_features(scenepoint2global_features)
        assert scenepoint2global_features.shape == (1, 1, self.n_feat_scenepoint2global_agg)

        # Concatenate features
        x = torch.cat([
            view2global_features.squeeze(0), # (1, n_feat_view2global_agg)
            scenepoint2global_features.squeeze(0), # (1, n_feat_scenepoint2global_agg)
        ], dim=1)
        assert x.shape == (1, self.n_feat_view2global_agg + self.n_feat_scenepoint2global_agg)

        if (self.n_feat_view2global_agg + self.n_feat_scenepoint2global_agg) != self.n_feat_global_out:
            x = self.proj_view_and_scenepoint2global(x)

        if prev_global_features is not None:
            x_skip = prev_global_features
            assert x.shape == x_skip.shape # NOTE: If False, we need to implement projection of skip connection to expected dimensionality.
            x = x_skip + x

        # Apply MLP
        x_skip = x
        if self.use_norm_pre_mlp:
            x = self.norm_pre_mlp(x)
            x = F.relu(x)
        x = self.mlp(x)
        assert x.shape == (1, self.n_feat_global_out)
        assert x_skip.ndim == 2 and x_skip.shape[1] == self.n_feat_global_out # NOTE: If False, we need to implement projection of skip connection to expected dimensionality.
        x = x_skip + x

        global_features = x

        return global_features

class Global2View(Module):
    def __init__(
        self,
        n_feat_global_in,
        n_feat_view_in_out,
        n_hidden_layers_view_update = 0,
        use_norm_global2view_update = True,
    ):
        super(Global2View, self).__init__()
        self.n_feat_global_in = n_feat_global_in
        self.n_feat_view_in_out = n_feat_view_in_out
        self.n_hidden_layers_view_update = n_hidden_layers_view_update
        self.use_norm_global2view_update = use_norm_global2view_update

        if self.use_norm_global2view_update:
            self.view_norm_layer = LayerNorm(n_feat_view_in_out)
            self.global_norm_layer = LayerNorm(n_feat_global_in)

        self.lin_view = Linear(n_feat_view_in_out, n_feat_view_in_out)
        self.lin_global = Linear(n_feat_global_in, n_feat_view_in_out, bias=False)

        if self.n_hidden_layers_view_update > 0:
            # NOTE: This self.mlp module will have one hidden layer less than specified.
            # This is due to that we conceptionally regard the above linear layers as part of the MLP.
            self.mlp = get_linear_layers((1 + self.n_hidden_layers_view_update - 1) * [n_feat_view_in_out] + [n_feat_view_in_out], init_activation=False, final_activation=False, norm=False)
        else:
            # The linear layers above are enough to achieve the desired 1-layer "MLP".
            pass

    def forward(self, global_features, prev_view_features):
        # The first MLP layer is implemented as separate projections for the different sources, followed by addition:
        # This is easier to implement and more efficient to compute, but equivalent to:
        #   1. Expand the global features across all views.
        #   2. Concatenate the features with the current features.
        #   3. Apply a linear projection element-wise on the resulting concatenated features.
        n_views = prev_view_features.shape[0]
        view_features = prev_view_features
        if self.use_norm_global2view_update:
            view_features = self.view_norm_layer(view_features)
            view_features = F.relu(view_features, inplace=True)
        view_features = self.lin_view(view_features)
        assert view_features.shape == (n_views, self.n_feat_view_in_out)
        if self.use_norm_global2view_update:
            global_features = self.global_norm_layer(global_features)
            global_features = F.relu(global_features, inplace=True)
        global_features = self.lin_global(global_features)
        assert global_features.shape == (1, self.n_feat_view_in_out)
        view_features = view_features + global_features
        assert view_features.shape == (n_views, self.n_feat_view_in_out)

        # If an MLP is desired (#hidden layers > 0), further layers should be added:
        if self.n_hidden_layers_view_update > 0:
            view_features = F.relu(view_features, inplace=True)
            view_features = self.mlp(view_features)

        view_features = prev_view_features + view_features

        return view_features

class Global2ScenePoint(Module):
    def __init__(
        self,
        n_feat_global_in,
        n_feat_scenepoint_in_out,
        n_hidden_layers_scenepoint_update = 0,
        use_norm_global2scenepoint_update = True,
    ):
        super(Global2ScenePoint, self).__init__()
        self.n_feat_global_in = n_feat_global_in
        self.n_feat_scenepoint_in_out = n_feat_scenepoint_in_out
        self.n_hidden_layers_scenepoint_update = n_hidden_layers_scenepoint_update
        self.use_norm_global2scenepoint_update = use_norm_global2scenepoint_update

        if self.use_norm_global2scenepoint_update:
            self.scenepoint_norm_layer = LayerNorm(n_feat_scenepoint_in_out)
            self.global_norm_layer = LayerNorm(n_feat_global_in)

        self.lin_scenepoint = Linear(n_feat_scenepoint_in_out, n_feat_scenepoint_in_out)
        self.lin_global = Linear(n_feat_global_in, n_feat_scenepoint_in_out, bias=False)

        if self.n_hidden_layers_scenepoint_update > 0:
            # NOTE: This self.mlp module will have one hidden layer less than specified.
            # This is due to that we conceptionally regard the above linear layers as part of the MLP.
            self.mlp = get_linear_layers((1 + self.n_hidden_layers_scenepoint_update - 1) * [n_feat_scenepoint_in_out] + [n_feat_scenepoint_in_out], init_activation=False, final_activation=False, norm=False)
        else:
            # The linear layers above are enough to achieve the desired 1-layer "MLP".
            pass

    def forward(self, global_features, prev_scenepoint_features):
        # The first MLP layer is implemented as separate projections for the different sources, followed by addition:
        # This is easier to implement and more efficient to compute, but equivalent to:
        #   1. Expand the global features across all scenepoints.
        #   2. Concatenate the features with the current features.
        #   3. Apply a linear projection element-wise on the resulting concatenated features.
        n_scenepoints = prev_scenepoint_features.shape[0]
        scenepoint_features = prev_scenepoint_features
        if self.use_norm_global2scenepoint_update:
            scenepoint_features = self.scenepoint_norm_layer(scenepoint_features)
            scenepoint_features = F.relu(scenepoint_features, inplace=True)
        scenepoint_features = self.lin_scenepoint(scenepoint_features)
        assert scenepoint_features.shape == (n_scenepoints, self.n_feat_scenepoint_in_out)
        if self.use_norm_global2scenepoint_update:
            global_features = self.global_norm_layer(global_features)
            global_features = F.relu(global_features, inplace=True)
        global_features = self.lin_global(global_features)
        assert global_features.shape == (1, self.n_feat_scenepoint_in_out)
        scenepoint_features = scenepoint_features + global_features
        assert scenepoint_features.shape == (n_scenepoints, self.n_feat_scenepoint_in_out)

        # If an MLP is desired (#hidden layers > 0), further layers should be added:
        if self.n_hidden_layers_scenepoint_update > 0:
            scenepoint_features = F.relu(scenepoint_features, inplace=True)
            scenepoint_features = self.mlp(scenepoint_features)

        scenepoint_features = prev_scenepoint_features + scenepoint_features

        return scenepoint_features

class GraphAttnSfMGlobalFeatureUpdate(Module):
    def __init__(
        self,
        n_feat_proj_in,
        n_feat_scenepoint_out,
        n_feat_view_out,
        n_feat_proj2scenepoint_agg = None,
        n_feat_proj2view_agg = None,
        n_feat_global_out = None,
        n_feat_scenepoint2global_agg = None,
        n_feat_view2global_agg = None,
        output_global = True,
        n_heads = 1,
        stateful = True,
        global2view_and_global2scenepoint_enabled = True,
        n_hidden_layers_scenepoint_update = 0,
        n_hidden_layers_view_update = 0,
        n_hidden_layers_global_update = 0,
    ):
        super(GraphAttnSfMGlobalFeatureUpdate, self).__init__()
        self.n_feat_proj_in = n_feat_proj_in
        self.n_feat_scenepoint_out = n_feat_scenepoint_out
        self.n_feat_view_out = n_feat_view_out
        self.n_feat_global_out = n_feat_global_out
        # self.n_feat_proj2scenepoint_agg = n_feat_proj2scenepoint_agg
        # self.n_feat_proj2view_agg = n_feat_proj2view_agg
        # self.n_feat_scenepoint2global_agg = n_feat_scenepoint2global_agg
        # self.n_feat_view2global_agg = n_feat_view2global_agg
        self.global2view_and_global2scenepoint_enabled = global2view_and_global2scenepoint_enabled

        if output_global or self.global2view_and_global2scenepoint_enabled:
            assert self.n_feat_global_out is not None

        # # n is the number of points and m is the number of cameras
        # self.lin_scenepoint = Linear(n_feat_proj_in, n_feat_scenepoint_out)
        # self.lin_view = Linear(n_feat_proj_in, n_feat_view_out)
        # self.lin_global = Linear(n_feat_proj_in, n_feat_global_out)

        assert n_feat_scenepoint_out % n_heads == 0
        assert n_feat_view_out % n_heads == 0
        self.proj2view = Proj2View(
            n_feat_proj_in,
            n_feat_view_out,
            n_heads,
            stateful = stateful,
            use_norm_pre_mlp = True,
            n_feat_proj2view_agg = n_feat_proj2view_agg,
            n_hidden_layers_view_update = n_hidden_layers_view_update,
        )
        self.proj2scenepoint = Proj2ScenePoint(
            n_feat_proj_in,
            n_feat_scenepoint_out,
            n_heads,
            stateful = stateful,
            use_norm_pre_mlp = True,
            n_feat_proj2scenepoint_agg = n_feat_proj2scenepoint_agg,
            n_hidden_layers_scenepoint_update = n_hidden_layers_scenepoint_update,
        )
        self.output_global = output_global
        if output_global or self.global2view_and_global2scenepoint_enabled:
            assert n_feat_global_out is not None
            assert n_feat_global_out % n_heads == 0
            self.view_and_scenepoint2global = ViewAndScenePoint2Global(
                n_feat_scenepoint_out,
                n_feat_view_out,
                n_feat_global_out,
                n_heads,
                stateful = stateful,
                use_norm_pre_mlp = True,
                n_feat_scenepoint2global_agg = n_feat_scenepoint2global_agg,
                n_feat_view2global_agg = n_feat_view2global_agg,
                n_hidden_layers_global_update = n_hidden_layers_global_update,
            )
        if self.global2view_and_global2scenepoint_enabled:
            self.global2view = Global2View(
                n_feat_global_out,
                n_feat_view_out,
                n_hidden_layers_view_update = n_hidden_layers_view_update,
                use_norm_global2view_update = True,
            )
            self.global2scenepoint = Global2ScenePoint(
                n_feat_global_out,
                n_feat_scenepoint_out,
                n_hidden_layers_scenepoint_update = n_hidden_layers_scenepoint_update,
                use_norm_global2scenepoint_update = True,
            )

    def forward(
        self,
        x, # projection features, normalized, to be fed as inputs to the "cross-attention" residual mappings for view features and scenepoint features, respectively.
        graph_structure,
        prev_scenepoint_features = None,
        prev_view_features = None,
        prev_global_features = None,
    ):
        # x is [m,n,d] sparse matrix
        m, n, n_feat_proj_in = x.shape
        assert n_feat_proj_in == self.n_feat_proj_in

        # Within this method, let's use a conventional pytorch sparse tensor, rather than the custom SparseMat.
        x = x.to_torch_hybrid_sparse_coo()

        scenepoint_features = self.proj2scenepoint(
            x,
            graph_structure['proj2scenepoint'],
            prev_scenepoint_features = prev_scenepoint_features,
        )
        # mean_colwise = sparse_utils.sparse_mean(x, dim=0).to_dense() # [m,n, n_feat_proj_in] -> [n, n_feat_proj_in]
        # scenepoint_features = self.lin_scenepoint(mean_colwise)  # [n, n_feat_proj_in] -> [n, n_feat_scenepoint_out]
        assert scenepoint_features.shape == (n, self.n_feat_scenepoint_out)

        view_features = self.proj2view(
            x,
            graph_structure['proj2view'],
            prev_view_features = prev_view_features,
        )
        # mean_rowwise = sparse_utils.sparse_mean(x, dim=1).to_dense() # [m,n, n_feat_proj_in] -> [m, n_feat_proj_in]
        # view_features = self.lin_view(mean_rowwise)  # [m, n_feat_proj_in] -> [m, n_feat_view_out]
        assert view_features.shape == (m, self.n_feat_view_out)

        if self.output_global or self.global2view_and_global2scenepoint_enabled:
            # NOTE: The most analogous way to replace the previous global mean (+ linear layer) with a graph convolution would be to use one graph convolutional layer and feed all projection features as source nodes.
            # Instead, we feed the previously computed view features and scenepoint features as source nodes, which is much more computationally feasible.
            # We use separate graph convolutions for the two sets of source nodes (with different parameters), and then merge them.
            global_features = self.view_and_scenepoint2global(
                view_features, # (m, n_feat_view_out)
                scenepoint_features, # (n, n_feat_scenepoint_out)
                graph_structure['view2global'],
                graph_structure['scenepoint2global'],
                prev_global_features = prev_global_features,
            )
            # global_features = self.lin_global(sparse_utils.sparse_mean(x, dim=(0, 1))[None, :].to_dense())  # [1, n_feat_proj_in] -> [1, n_feat_global_out]
            assert global_features.shape == (1, self.n_feat_global_out)

        if self.global2view_and_global2scenepoint_enabled:
            scenepoint_features = self.global2scenepoint(
                global_features,
                scenepoint_features,
            )
            view_features = self.global2view(
                global_features,
                view_features,
            )

        if not self.output_global:
            return scenepoint_features, view_features
        else:
            return scenepoint_features, view_features, global_features


class GraphAttnSfMProjectionFeatureUpdate(Module):
    def __init__(
        self,
        n_feat_proj_in,
        n_feat_scenepoint_in,
        n_feat_view_in,
        n_feat_global_in,
        n_feat_proj_out,
        n_hidden_layers_proj_update = 0,
        normalize_global_features = True,
    ):
        super(GraphAttnSfMProjectionFeatureUpdate, self).__init__()
        self.n_feat_proj_in = n_feat_proj_in
        self.n_feat_scenepoint_in = n_feat_scenepoint_in
        self.n_feat_view_in = n_feat_view_in
        self.n_feat_global_in = n_feat_global_in
        self.n_feat_proj_out = n_feat_proj_out
        self.n_hidden_layers_proj_update = n_hidden_layers_proj_update
        self.normalize_global_features = normalize_global_features

        if self.normalize_global_features:
            self.scenepoint_norm_layer = LayerNorm(n_feat_scenepoint_in)
            self.view_norm_layer = LayerNorm(n_feat_view_in)
            self.global_norm_layer = LayerNorm(n_feat_global_in)

        self.lin_proj = Linear(n_feat_proj_in, n_feat_proj_out)
        self.lin_scenepoint = Linear(n_feat_scenepoint_in, n_feat_proj_out, bias=False)
        self.lin_view = Linear(n_feat_view_in, n_feat_proj_out, bias=False)
        self.lin_global = Linear(n_feat_global_in, n_feat_proj_out, bias=False)

        if self.n_hidden_layers_proj_update > 0:
            # NOTE: This self.mlp module will have one hidden layer less than specified.
            # This is due to that we conceptionally regard the above linear layers as part of the MLP.
            self.mlp = get_linear_layers((1 + self.n_hidden_layers_proj_update - 1) * [n_feat_proj_out] + [n_feat_proj_out], init_activation=False, final_activation=False, norm=False)
        else:
            # The linear layers above are enough to achieve the desired 1-layer "MLP".
            pass

    def forward(
        self,
        scenepoint_features,
        view_features,
        global_features,
        x, # projection features, expected to be normalized already.
    ):
        # x is [m,n,d] sparse matrix

        # Within this method, let's use a conventional pytorch sparse tensor, rather than the custom SparseMat.
        cam_per_pts, pts_per_cam = x.cam_per_pts, x.pts_per_cam
        x = x.to_torch_hybrid_sparse_coo()
        nnz = x.values().shape[0]

        proj_features = x.values()

        if self.normalize_global_features:
            # NOTE: Projection features already normalized
            scenepoint_features = self.scenepoint_norm_layer(scenepoint_features)
            scenepoint_features = F.relu(scenepoint_features, inplace=True)
            view_features = self.view_norm_layer(view_features)
            view_features = F.relu(view_features, inplace=True)
            global_features = self.global_norm_layer(global_features)
            global_features = F.relu(global_features, inplace=True)

        # The first MLP layer is implemented as separate projections for the different sources, followed by addition:
        # This is easier to implement and more efficient to compute, but equivalent to:
        #   1. Expand the global features across the sparse matrix.
        #   2. Concatenate the features from all sources.
        #   3. Apply a linear projection element-wise on the resulting concatenated features.
        proj_features = self.lin_proj(proj_features)  # [nnz, n_feat_proj_in] -> [nnz,n_feat_proj_out]
        scenepoint_features = self.lin_scenepoint(scenepoint_features)
        view_features = self.lin_view(view_features)
        global_features = self.lin_global(global_features)
        new_features = (proj_features + scenepoint_features[x.indices()[1], :] + view_features[x.indices()[0], :] + global_features) / 4  # [nnz,n_feat_proj_out]
        assert new_features.shape == (nnz, self.n_feat_proj_out)

        # If an MLP is desired (#hidden layers > 0), further layers should be added:
        if self.n_hidden_layers_proj_update > 0:
            new_features = F.relu(new_features, inplace=True)
            new_features = self.mlp(new_features)

        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        # NOTE: Importantly, we do not mix values / indices of the new tensor with those of the input SparseMat, as the order may have changed due to .coalesce().
        return SparseMat(new_features, x.indices(), cam_per_pts, pts_per_cam, new_shape)


class ProjLayer(Module):
    def __init__(self, n_feat_proj_in, n_feat_proj_out):
        super(ProjLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_proj = Linear(n_feat_proj_in, n_feat_proj_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_proj(x.values)  # [nnz, n_feat_proj_in] -> [nnz,n_feat_proj_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


def normalize_projection_features(x, norm_layer=None):
    features = x.values
    if norm_layer is not None:
        norm_features = norm_layer(features)
    else:
        norm_features = features - features.mean(dim=0, keepdim=True)
        # norm_features = norm_features / norm_features.std(dim=0, keepdim=True)
    return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


def relu_on_projection_features(x):
    new_features = F.relu(x.values, inplace=True)
    return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class IdentityLayer(Module):
    def forward(self, x):
        return x


class EmbeddingLayer(Module):
    def __init__(self, pos_emb_n_freq, in_dim, post_embed_proj_dim=None):
        super(EmbeddingLayer, self).__init__()
        if pos_emb_n_freq > 0:
            self.embed, self.d_out = get_embedder(pos_emb_n_freq, in_dim)
        else:
            self.embed, self.d_out = (Identity(), in_dim)
        self.post_embed_proj_dim = post_embed_proj_dim
        if self.post_embed_proj_dim is not None:
            if post_embed_proj_dim == -1:
                # Convenience option for preserving the dimensionality of the embedding post-projection.
                post_embed_proj_dim = self.d_out
            self.post_embed_lin = Linear(self.d_out, post_embed_proj_dim)
            self.d_out = post_embed_proj_dim
        else:
            self.post_embed_lin = None

    def forward(self, x):
        # (m, n, n_feat_in) -> (m, n, n_feat_embedding)
        embeded_features = self.embed(x.values)
        if self.post_embed_lin is not None:
            embeded_features = self.post_embed_lin(embeded_features)
        new_shape = (x.shape[0], x.shape[1], embeded_features.shape[1])
        return SparseMat(embeded_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)
