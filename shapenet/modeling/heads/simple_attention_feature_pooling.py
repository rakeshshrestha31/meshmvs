##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import torch
import torch.nn as nn

class SimpleAttentionFeaturePooling(nn.Module):
    def __init__(self, input_features_dim, bias=False):
        super(SimpleAttentionFeaturePooling, self).__init__()
        self.linear = nn.Linear(input_features_dim, 1, bias=bias)
        self.softmax = nn.Softmax(dim=0)

    ##
    #  @param features tensor of dimensions views x batch x features
    #  @return attention weighted features of size batch x features
    def forward(self, features):
        num_views, batch_size, num_input_features = features.size()
        flattened_features = features.view(-1, features.size(-1))
        weights = self.linear(flattened_features)
        weights = weights.view(num_views, batch_size, 1)
        weights = self.softmax(weights)
        weights_expanded = weights.expand(-1, -1, num_input_features)
        weighted_sum = (features * weights_expanded).sum(dim=0)
        return weighted_sum, weights
