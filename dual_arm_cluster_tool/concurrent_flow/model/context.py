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

        # Stack loadlock1_wafer_embedding by state.loadlock1_wafer_in
        """
        loadport_size = 25
        loadlock1_wafer_embeddings = loadlock1_wafer_embedding[:, None, :].repeat(1, 25, 1)
        mask = torch.arange(loadport_size)[None, :, None].repeat(state.batch_size(), 1, 1).to(state.device()) \
            < state.loadlock1_wafer_in[:, :, None]

        loadlock1_wafer_embeddings = (loadlock1_wafer_embeddings * mask).sum(dim=1)

        # Stack loadlock2_wafer_embedding by state.loadlock2_wafer_in
        loadlock2_wafer_embeddings = loadlock2_wafer_embedding[:, None, :].repeat(1, 25, 1)
        mask = torch.arange(loadport_size)[None, :, None].repeat(state.batch_size(), 1, 1).to(state.device()) \
            < state.loadlock2_wafer_in[:, :, None]

        loadlock2_wafer_embeddings = (loadlock2_wafer_embeddings * mask).sum(dim=1)
        """

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


class single_armed_robot_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(single_armed_robot_context, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)

    def forward(self, pm_embedding, state):
        """
        robot loc embedding = [robot located PM contexutalized embedding]
        """
        loc_embedding = torch.cat(
            [torch.zeros_like(pm_embedding[:, 0:1]), pm_embedding], dim=1)

        loc_embedding = torch.cat(
            [loc_embedding, torch.ones_like(pm_embedding[:, 0:1])], dim=1)

        robot_loc_embedding = gather_by_index(loc_embedding, state.robot_loc)
        robot_context = self.linear(robot_loc_embedding)

        return robot_context


class dual_armed_robot_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(dual_armed_robot_context, self).__init__()
        self.linear = nn.Linear(embedding_dim*2, embedding_dim, bias=linear_bias)

    def forward(self, arm1_embeddings, arm2_embeddings):
        # robot loc embedding
        robot_context = self.linear(torch.cat([arm1_embeddings, arm2_embeddings], dim=-1))

        return robot_context

