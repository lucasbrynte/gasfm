import torch
from torch import nn
from models.baseNet import BaseNet
from models.layers import *


class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.proj_feat_normalization = conf.get_bool("model.proj_feat_normalization")
        self.add_skipconn_for_residual_blocks = conf.get_bool("model.add_skipconn_for_residual_blocks")

        self.layers = []
        self.layers.append(SetOfSetLayer(d_in, d_out))
        for i in range(1, self.block_size):
            self.layers.append(SetOfSetLayer(d_out, d_out))
        self.layers = torch.nn.ModuleList(self.layers)

        if self.add_skipconn_for_residual_blocks:
            if d_in == d_out:
                self.skip_projection = None
            else:
                self.skip_projection = ProjLayer(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix

        xl = x
        for i, layer in enumerate(self.layers):
            xl = layer(xl)
            if i < len(self.layers) - 1:
                if self.proj_feat_normalization:
                    xl = normalize_projection_features(xl)
                xl = relu_on_projection_features(xl)

        if self.add_skipconn_for_residual_blocks:
            x_skip = x
            if self.skip_projection is not None:
                x_skip = self.skip_projection(x_skip)
                if self.proj_feat_normalization:
                    x_skip = normalize_projection_features(x_skip)
            xl = x_skip + xl

        out = relu_on_projection_features(xl)
        return out


class SetOfSetNet(BaseNet):
    def __init__(self, conf, batchnorm=False):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        pos_emb_n_freq = conf.get_int('model.pos_emb_n_freq')
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

        self.embed = EmbeddingLayer(pos_emb_n_freq, d_in)

        self.equivariant_blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.equivariant_blocks.append(SetOfSetBlock(
                self.embed.d_out if i == 0 else num_feats,
                n_feat_proj_depth_head if self.depth_head_enabled and (i == num_blocks - 1) else num_feats,
                conf,
            ))

        if self.view_head_enabled or self.scenepoint_head_enabled:
            if not self.view_head_enabled and self.scenepoint_head_enabled:
                raise NotImplementedError('Final feature aggregation for only view features or scenepoint features alone is not implemented.')
            self.final_global_update = SetOfSetGlobalFeatureUpdate(num_feats, num_feats, output_global=False)
        if self.batchnorm:
            raise NotImplementedError()
        if self.depth_head_enabled:
            self.depth_head = get_linear_layers((1 + n_hidden_layers_depth_head) * [n_feat_proj_depth_head] + [depth_d_out], init_activation=False, final_activation=False, norm=False)
        if self.view_head_enabled:
            self.view_head = get_linear_layers((1 + n_hidden_layers_view_head) * [num_feats] + [view_d_out], init_activation=False, final_activation=False, norm=False)
            # self.view_head = get_linear_layers([num_feats] * 2 + [view_d_out], init_activation=False, final_activation=False, norm=False)
        if self.scenepoint_head_enabled:
            self.scenepoint_head = get_linear_layers((1 + n_hidden_layers_scenepoint_head) * [num_feats] + [scenepoint_d_out], init_activation=False, final_activation=False, norm=False)
            # self.scenepoint_head = get_linear_layers([num_feats] * 2 + [scenepoint_d_out], init_activation=False, final_activation=False, norm=False)

    def forward(self, data):
        x = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        if self.view_head_enabled or self.scenepoint_head_enabled:
            # Final global aggregation / feature update
            n_input, m_input = self.final_global_update(x)
            if self.batchnorm:
                raise NotImplementedError()
            m_input = nn.functional.relu(m_input)
            n_input = nn.functional.relu(n_input)

        if self.depth_head_enabled:
            # Depth regression
            n_views, n_scenepoints = x.shape[:2]
            depth_out = SparseMat(self.depth_head(x.values), x.indices, x.cam_per_pts, x.pts_per_cam, [n_views, n_scenepoints, 1])

        if self.view_head_enabled:
            # Cameras predictions
            # m_input = x.mean(dim=1) # [m,d_out]
            m_out = self.view_head(m_input)  # [m, d_m]

        if self.scenepoint_head_enabled:
            # Points predictions
            # n_input = x.mean(dim=0) # [n,d_out]
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
