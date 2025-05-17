import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.CONCAT.utils import gather_by_index

def get_init_embedding(**params):
    EMBEDDING_REGISTRY = {
        'single': LotStageEmbedding,
        'dual': LotStageEmbedding
    }

    arm_type = params['arm_type']

    init_embed = EMBEDDING_REGISTRY.get(arm_type, None)
    if init_embed is None:
        raise ValueError(f"Unknown arm type {arm_type}.\
                         Available arms: {EMBEDDING_REGISTRY.keys()}")

    return init_embed(**params)

class LotStageEmbedding(nn.Module):
    def __init__(self, **params):
        super(LotStageEmbedding, self).__init__()
        self.device = params['device']

        self.row_cnt = params['num_lot_type']
        self.col_cnt = len(params['stage'])

        embedding_dim = params['embedding_dim']
        self.purge_constraint = params['max_purge_time'] != 0
        row_feat_dim = 1 # {#wafer} -> new: {0}
        #col_feat_dim = self.col_cnt + 2 if self.purge_constraint else self.col_cnt+1 # {order, #PM, purge time}
        col_feat_dim = self.col_cnt + 2 # ={order, #PM, purge time}

        self.proj_row_feat = nn.Linear(row_feat_dim, embedding_dim, bias=False)
        self.proj_col_feat = nn.Linear(col_feat_dim, embedding_dim, bias=False)
        self.to(self.device)


    def forward(self, env, state):
        batch_size = state.batch_size()

        # row(lot) embedding
        # {zero}: R->R^d
        # ------------------------------------------------------
        row_feat = torch.zeros(size=(batch_size, self.row_cnt, 1), device=self.device)
        row_emb = self.proj_row_feat(row_feat)

        # col(stage) embedding
        # {Stage order, # PM}: R^(num_stage[one-hot]+1) -> R^d
        # ------------------------------------------------------
        # feat 1: stage order
        stage_order = F.one_hot(
            torch.arange(self.col_cnt),num_classes=self.col_cnt
            )[None, :, :].repeat(batch_size, 1, 1).to(self.device)
        # (batch, num_stage), one-hot-encoding

        # feat 2
        stage_capacity = (torch.tensor(env.stage) / max(env.stage)
                          )[None, :, None].repeat(batch_size, 1, 1).to(self.device)

        # feat 3
        if self.purge_constraint: stage_purge_time = env.loc.purge_time[:, 1:-1, None].to(self.device)
        else: stage_purge_time = torch.zeros_like(stage_capacity) # zero purge time

        col_feat = torch.cat([stage_order, stage_capacity, stage_purge_time], dim=-1)
        col_emb = self.proj_col_feat(col_feat)

        # mat feature
        # ------------------------------------------------------
        process_time = env.recipe_table['process_time'][:, :, 1:self.col_cnt+1, None].to(self.device)
        cost_mat = process_time
        # shape: (batch, row, col, 1)

        return row_emb, col_emb, cost_mat

