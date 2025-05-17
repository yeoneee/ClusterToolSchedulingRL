import math
import torch
import torch.nn as nn
from model.CONCAT.utils import gather_by_index


# Context list
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        batch_size, seq_len, embedding_dim = x.size()

        pe = self.pe[:seq_len].squeeze(1)[None, :].repeat(batch_size, 1, 1)
        #pe_order = gather_by_index(pe, order)
        #x = x + pe_order
        x = x + pe
        return self.dropout(x)


class EfficientPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(EfficientPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(x.size(1), self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand_as(x)
        return x + pe

class loadport_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(loadport_context, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)
        #self.context_linear = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)

    def forward(self, env, encoded_row, state):

        # get loadport wafers
        exit_foup = torch.clamp(env.wafer.exit_foup, max=env.num_foup-1)
        wafer_types = env.wafer.recipe[env.batch_idx, exit_foup].to(state.device())
        wafer_embeddings = gather_by_index(encoded_row, wafer_types)
        if wafer_embeddings.ndim == 2:
            wafer_embeddings = wafer_embeddings[None, :, :]

        # mask already inserted
        loadport_wafers = (env.wafer.loc[env.batch_idx, exit_foup] == 0).unsqueeze(-1).to(state.device())
        masked_loadport_wafer_embeddings = wafer_embeddings * loadport_wafers

        # pooling embeddings
        ll_context = self.linear(masked_loadport_wafer_embeddings.sum(dim=1))
        """
        wafer_types = env.wafer.recipe.to(state.device())
        wafer_embeddings = gather_by_index(encoded_row, wafer_types.reshape(*env.batch_size, -1))
        loadport_wafers = (env.wafer.loc == 0).reshape(*env.batch_size, -1).to(state.device()).unsqueeze(-1)
        masked_loadport_wafer_embeddings = wafer_embeddings * loadport_wafers
        ll_context = self.linear(masked_loadport_wafer_embeddings.sum(dim=1))
        """

        return ll_context


class pm_context(nn.Module):
    def __init__(self, env, embedding_dim, linear_bias=False):
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
        # Example: PM1, 2, 3, 4, outLL(=5)
                   robot loc = 5
                   then PM is dummy embeddings with index 4 (=robot loc -1)
        """
        pm_embedding_add_dummy = torch.cat(
            [pm_embedding, torch.zeros_like(pm_embedding[:, 0:1])], dim=1)
        robot_pm_embedding = pm_embedding_add_dummy[state.batch_idx, state.robot_loc-1]
        robot_context = self.linear(robot_pm_embedding)

        return robot_context


class dual_armed_robot_context(nn.Module):
    def __init__(self, embedding_dim, linear_bias=False):
        super(dual_armed_robot_context, self).__init__()
        """
        """
        self.linear = nn.Linear(embedding_dim*2, embedding_dim, bias=linear_bias)

    def forward(self, env, encoded_row, encoded_col, state):
        """
        robot hold wafer embedding + wafer's next stage embedding
        """

        # robot hold wafer embedding
        # ---------------------------------------------------------------------------
        encoded_row_add_dummy = torch.cat([encoded_row, torch.zeros_like(encoded_row[:, 0:1])], dim=1)
        robot_lot_idx = env.wafer.get_recipe(state.robot_hold_wafer) # shape (batch, 2)
        robot_lot_dummy_idx = torch.where(robot_lot_idx <= env.num_lot_type, robot_lot_idx, encoded_row_add_dummy.size(1)-1)

        robot_lot_embed = gather_by_index(encoded_row_add_dummy, robot_lot_dummy_idx)
        if robot_lot_embed.ndim == 2:
            robot_lot_embed = robot_lot_embed[None, :, :]
        # (batch, 2, embedding)

        # robot hold wafer next stage embedding
        # ---------------------------------------------------------------------------
        robot_lot_next_step = env.wafer.get_step(state.robot_hold_wafer) + 1
        dummy_wafer_idx = robot_lot_next_step > env.num_step

        robot_lot_dummy_next_step = torch.where(~dummy_wafer_idx, robot_lot_next_step, 0)
        robot_lot_dummy_idx = torch.where(robot_lot_idx <= env.num_lot_type, robot_lot_idx, 0)

        robot_lot_next_stage = env.recipe_table.get('flow').to(state.device())\
            [state.batch_idx[:, None].repeat(1,2), robot_lot_dummy_idx, robot_lot_dummy_next_step]
        encoded_col_add_dummy = torch.cat([torch.zeros_like(encoded_col[:, 0:1]), encoded_col], dim=1)
        encoded_col_add_dummy = torch.cat([encoded_col_add_dummy, torch.zeros_like(encoded_col[:, 0:1])], dim=1)
        robot_next_stage_embed = gather_by_index(encoded_col_add_dummy, robot_lot_next_stage) # (batch, 2, embedding)
        if robot_next_stage_embed.ndim == 2:
            robot_next_stage_embed = robot_next_stage_embed[None, :, :]
        robot_next_stage_embed[dummy_wafer_idx[:, :, None].repeat(1,1,encoded_col.size(-1))] = 0.

        robot_embedding = robot_lot_embed + robot_next_stage_embed
        robot_embedding = robot_embedding.reshape(state.batch_size(), -1)
        robot_context = self.linear(robot_embedding)

        return robot_context

