import math
import torch
import torch.nn as nn
from model.utils import gather_by_index


# Context list
# ---------------------------------------------------------------------------
class loadport_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(loadport_context, self).__init__()
        self.ratio_linear = nn.Linear(1, embedding_dim, bias=linear_bias)
        self.linear = nn.Linear(embedding_dim*3, embedding_dim, bias=linear_bias)
        #self.linear = nn.Linear(embedding_dim*2, embedding_dim, bias=linear_bias)

    def forward(self, encoded_row, state):
        # get loadport wafers
        loadlock1_wafer_embedding = gather_by_index(encoded_row, state.loadlock1_wafer_recipe)
        loadlock2_wafer_embedding = gather_by_index(encoded_row, state.loadlock2_wafer_recipe)
        ratio_embedding = self.ratio_linear(state.loadlock1_wafer_in / state.loadlock2_wafer_in)
        # pooling embeddings
        ll_context = self.linear(torch.cat([loadlock1_wafer_embedding, loadlock2_wafer_embedding, ratio_embedding], dim=-1))
        #ll_context = self.linear(torch.cat([loadlock1_wafer_embedding, loadlock2_wafer_embedding], dim=-1))

        return ll_context


class pm_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(pm_context, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)

    def forward(self, pm_embeddings):
        q = self.Wq(pm_embeddings)
        k = self.Wk(pm_embeddings)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(pm_embeddings.size(-1))
        weights = torch.softmax(scores, dim=-1)
        self_attn_pm_embeddings = weights @ pm_embeddings
        pm_context = self.linear(self_attn_pm_embeddings.sum(dim=1))

        return pm_context


class dual_armed_robot_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(dual_armed_robot_context, self).__init__()
        self.linear = nn.Linear(embedding_dim*2, embedding_dim, bias=linear_bias)

    def forward(self, arm1_embeddings, arm2_embeddings):
        # robot loc embedding
        robot_context = self.linear(torch.cat([arm1_embeddings, arm2_embeddings], dim=-1))

        return robot_context

