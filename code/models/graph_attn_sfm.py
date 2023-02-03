import torch
from torch import nn
from models.baseNet import BaseNet
from models.layers import *
from utils import sparse_utils


class GraphAttnSfMNet(BaseNet):
    def __init__(self, conf, batchnorm=False):
        super(GraphAttnSfMNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_layers = conf.get_int('model.num_layers')
        n_heads = conf.get_int("model.n_heads")
        n_feat_proj = conf.get_int('model.n_feat_proj')
        n_feat_scenepoint = conf.get_int('model.n_feat_scenepoint')
        n_feat_view = conf.get_int('model.n_feat_view')
        n_feat_global = conf.get_int('model.n_feat_global')
        n_feat_proj2scenepoint_agg = conf.get_int('model.n_feat_proj2scenepoint_agg', default=None)
        n_feat_proj2view_agg = conf.get_int('model.n_feat_proj2view_agg', default=None)
        n_feat_scenepoint2global_agg = conf.get_int('model.n_feat_scenepoint2global_agg', default=None)
        n_feat_view2global_agg = conf.get_int('model.n_feat_view2global_agg', default=None)
        n_hidden_layers_scenepoint_update = conf.get_int("model.n_hidden_layers_scenepoint_update")
        n_hidden_layers_view_update = conf.get_int("model.n_hidden_layers_view_update")
        n_hidden_layers_global_update = conf.get_int("model.n_hidden_layers_global_update")
        n_hidden_layers_proj_update = conf.get_int("model.n_hidden_layers_proj_update")
        pos_emb_n_freq = conf.get_int('model.pos_emb_n_freq')
        use_norm_proj_update = conf.get_bool("model.use_norm_proj_update")
        add_residual_skipconn_proj_update = conf.get_bool("model.add_residual_skipconn_proj_update")
        self.add_skipconn_from_init_projfeat = conf.get_bool("model.add_skipconn_from_init_projfeat")
        self.stateful_global_features = conf.get_bool("model.stateful_global_features")
        global2view_and_global2scenepoint_enabled = conf.get_bool("model.global2view_and_global2scenepoint_enabled")
        self.depth_head_enabled = conf.get_bool('model.depth_head.enabled', default=False)
        self.view_head_enabled = conf.get_bool('model.view_head.enabled', default=False)
        self.scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled', default=False)
        if self.depth_head_enabled:
            n_feat_proj_depth_head = conf.get_int("model.depth_head.n_feat")
            n_hidden_layers_depth_head = conf.get_int('model.depth_head.n_hidden_layers')
        if self.view_head_enabled:
            n_hidden_layers_view_head = conf.get_int('model.view_head.n_hidden_layers')
        if self.scenepoint_head_enabled:
            n_hidden_layers_scenepoint_head = conf.get_int('model.scenepoint_head.n_hidden_layers')

        self.batchnorm = batchnorm

        if self.depth_head_enabled:
            depth_d_out = 1
        if self.scenepoint_head_enabled:
            scenepoint_d_out = 3
        if self.view_head_enabled:
            view_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(pos_emb_n_freq, d_in, post_embed_proj_dim=-1)
        d_emb = self.embed.d_out

        if self.add_skipconn_from_init_projfeat:
            self.n_feat_skipconn_init_projfeat_in = d_emb
        else:
            self.n_feat_skipconn_init_projfeat_in = 0

        self.equivariant_blocks = torch.nn.ModuleList()
        for i in range(num_layers):
            first_block = i == 0
            self.equivariant_blocks.append(GraphAttnSfMLayer(
                d_emb if i == 0 else n_feat_proj, # n_feat_proj_in
                n_feat_proj_depth_head if self.depth_head_enabled and (i == num_layers - 1) else n_feat_proj, # n_feat_proj_out
                n_feat_scenepoint, # n_feat_scenepoint_hidden
                n_feat_view, # n_feat_view_hidden
                n_feat_global, # n_feat_global_hidden
                n_feat_proj2scenepoint_agg = n_feat_proj2scenepoint_agg,
                n_feat_proj2view_agg = n_feat_proj2view_agg,
                n_feat_scenepoint2global_agg = n_feat_scenepoint2global_agg,
                n_feat_view2global_agg = n_feat_view2global_agg,
                use_norm_proj_update = use_norm_proj_update,
                add_residual_skipconn_proj_update = add_residual_skipconn_proj_update,
                n_feat_skipconn_init_projfeat_in = self.n_feat_skipconn_init_projfeat_in if not first_block and self.add_skipconn_from_init_projfeat else None,
                n_heads = n_heads,
                stateful = False if first_block else self.stateful_global_features,
                global2view_and_global2scenepoint_enabled = global2view_and_global2scenepoint_enabled,
                n_hidden_layers_scenepoint_update = n_hidden_layers_scenepoint_update,
                n_hidden_layers_view_update = n_hidden_layers_view_update,
                n_hidden_layers_global_update = n_hidden_layers_global_update,
                n_hidden_layers_proj_update = n_hidden_layers_proj_update,
            ))

        if self.view_head_enabled or self.scenepoint_head_enabled:
            if not self.view_head_enabled and self.scenepoint_head_enabled:
                raise NotImplementedError('Final feature aggregation for only view features or scenepoint features alone is not implemented.')
            self.final_global_update = GraphAttnSfMGlobalFeatureUpdate(
                n_feat_proj_depth_head if self.depth_head_enabled and (i == num_layers - 1) else n_feat_proj,
                n_feat_scenepoint,
                n_feat_view,
                n_feat_proj2scenepoint_agg = n_feat_proj2scenepoint_agg,
                n_feat_proj2view_agg = n_feat_proj2view_agg,
                n_feat_global_out = n_feat_global,
                n_feat_scenepoint2global_agg = n_feat_scenepoint2global_agg,
                n_feat_view2global_agg = n_feat_view2global_agg,
                output_global = False,
                n_heads = n_heads,
                stateful = self.stateful_global_features,
                global2view_and_global2scenepoint_enabled = global2view_and_global2scenepoint_enabled,
                n_hidden_layers_scenepoint_update = n_hidden_layers_scenepoint_update,
                n_hidden_layers_view_update = n_hidden_layers_view_update,
                n_hidden_layers_global_update = n_hidden_layers_global_update,
            )
        if self.batchnorm:
            raise NotImplementedError()
        if self.depth_head_enabled:
            self.depth_head = get_linear_layers((1 + n_hidden_layers_depth_head) * [n_feat_proj_depth_head] + [depth_d_out], init_activation=False, final_activation=False, norm=False)
        if self.view_head_enabled:
            self.view_head = get_linear_layers((1 + n_hidden_layers_view_head) * [n_feat_view] + [view_d_out], init_activation=False, final_activation=False, norm=False)
            # self.view_head = get_linear_layers([n_feat_view] * 2 + [view_d_out], init_activation=False, final_activation=False, norm=False)
        if self.scenepoint_head_enabled:
            self.scenepoint_head = get_linear_layers((1 + n_hidden_layers_scenepoint_head) * [n_feat_scenepoint] + [scenepoint_d_out], init_activation=False, final_activation=False, norm=False)
            # self.scenepoint_head = get_linear_layers([n_feat_scenepoint] * 2 + [scenepoint_d_out], init_activation=False, final_activation=False, norm=False)

    def forward(self, data):
        projection_features = data.x  # x is [m,n,d] sparse matrix
        # The graph structure depends on and is retrieved from the scene data:
        graph_structure = data.graph_wrappers
        projection_features = self.embed(projection_features)
        if self.add_skipconn_from_init_projfeat:
            skipconn_init_projfeat = projection_features
        else:
            skipconn_init_projfeat = None
        scenepoint_features = None
        view_features = None
        global_features = None
        for eq_block in self.equivariant_blocks:
            projection_features, scenepoint_features, view_features, global_features = eq_block(
                projection_features,
                graph_structure,
                prev_scenepoint_features = scenepoint_features if self.stateful_global_features else None,
                prev_view_features = view_features if self.stateful_global_features else None,
                prev_global_features = global_features if self.stateful_global_features else None,
                skipconn_init_projfeat = skipconn_init_projfeat if self.add_skipconn_from_init_projfeat else None,
            )  # [m,n,d_emb] -> [m,n,n_feat_proj]

        if self.view_head_enabled or self.scenepoint_head_enabled:
            # Final global aggregation / feature update
            n_input, m_input = self.final_global_update(
                projection_features,
                graph_structure,
                prev_scenepoint_features = scenepoint_features if self.stateful_global_features else None,
                prev_view_features = view_features if self.stateful_global_features else None,
                prev_global_features = global_features if self.stateful_global_features else None,
            )
            if self.batchnorm:
                raise NotImplementedError()
            m_input = nn.functional.relu(m_input)
            n_input = nn.functional.relu(n_input)

        if self.depth_head_enabled:
            # Depth regression
            n_views, n_scenepoints = projection_features.shape[:2]
            depth_out = SparseMat(
                self.depth_head(projection_features.values),
                projection_features.indices,
                projection_features.cam_per_pts,
                projection_features.pts_per_cam,
                [n_views, n_scenepoints, 1],
            )

        if self.view_head_enabled:
            # Cameras predictions
            # m_input = projection_features.mean(dim=1) # [m,n_feat_proj]
            m_out = self.view_head(m_input)  # [m, d_m]

        if self.scenepoint_head_enabled:
            # Points predictions
            # n_input = projection_features.mean(dim=0) # [n,n_feat_proj]
            n_out = self.scenepoint_head(n_input).T  # [n, d_n] -> [d_n, n]

        pred_dict = {}
        if self.depth_head_enabled:
            pred_depths_dict = self.extract_depth_outputs(depth_out)
            pred_dict.update(pred_depths_dict)
        if self.view_head_enabled:
            pred_views_dict = self.extract_view_outputs(m_out)
            pred_dict.update(pred_views_dict)
        if self.scenepoint_head_enabled:
            pred_scenepoints_dict = self.extract_scenepoint_outputs(n_out)
            pred_dict.update(pred_scenepoints_dict)

        return pred_dict
