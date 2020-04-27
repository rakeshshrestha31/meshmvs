##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import torch
import torch.nn as nn
from .multihead_attention import MultiheadAttention

class MultiHeadAttentionFeaturePooling(nn.Module):
    def __init__(self, input_features_dim, output_features_dim,
                 num_heads, use_stats_query, bias=True):
        super(MultiHeadAttentionFeaturePooling, self).__init__()
        self.use_stats_query = use_stats_query

        self.attention = MultiheadAttention(
            output_features_dim, num_heads, bias=bias
        )

        # linear modules ot transform input features to key/value/query
        self.query_linear = nn.Linear(
            input_features_dim, output_features_dim, bias=bias
        )
        self.key_linear = nn.Linear(
            input_features_dim, output_features_dim, bias=bias
        )
        self.value_linear = nn.Linear(
            input_features_dim, output_features_dim, bias=bias
        )

    ##
    #  @param features tensor of dimensions views x batch x features
    def forward(self, features):
        num_views, batch_size, num_input_features = features.size()
        flattened_features = features.view(-1, features.size(-1))

        # query
        if self.use_stats_query:
            flattened_query_input = self.get_features_stats(features) \
                                        .view(-1, features.size(-1))
        else:
            flattened_query_input = flattened_features

        query = self.query_linear(flattened_query_input)
        query = query.view(num_views, batch_size, -1)
        if not self.use_stats_query:
            # the first dim of query should be target sequence size (here 1)
            # taking mean is non-standard (TODO: find a better way)
            query = query.mean(dim=0, keepdim=True)

        # key/value
        key = self.key_linear(flattened_features)
        value = self.value_linear(flattened_features)

        attn_output, attn_weights = self.attention(query, key, value)
        return attn_output, attn_weights

    ##
    #  @param features tensor of dimensions views x batch x features
    @staticmethod
    def get_features_stats(features):
        max_features = torch.max(features, dim=0)[0]
        mean_features = torch.mean(features, dim=0)
        var_features = torch.var(features, dim=0, unbiased=False)
        # calculating std using torch methods give NaN gradients
        # var will have different unit that mean/max, hence std desired
        std_features = torch.sqrt(var_features + 1e-8)
        return torch.stack((max_features, mean_features, std_features), dim=0)

