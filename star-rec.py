import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import math
import torch.nn.functional as F

class STARRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(STARRec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        
        #Hyperparameters for MHAttention
        self.n_heads = config["n_heads"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        
        self.attention = MultiHeadAttention(
            n_heads = self.n_heads,
            hidden_size = self.hidden_size,
            hidden_dropout_prob = self.dropout_prob,
            attn_dropout_prob = self.dropout_prob,
            layer_norm_eps = self.layer_norm_eps,
        )
        self.weights_attn = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.MOELayer = MoE(d_model=self.hidden_size, n_experts=4, expert_hidden_size=4*self.hidden_size, dropout=self.dropout_prob)
        

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        
        for i in range(self.num_layers):
            mamba = self.mamba_layers[i](item_emb)
        
        attn_enhancement = self.attention(item_emb)

        weights_attn = F.softmax(self.weights_attn, dim=0)
        item_emb = weights_attn[0] * attn_enhancement + weights_attn[1] * mamba
        item_emb = self.MOELayer(item_emb)
        
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    
class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
    
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.gate = nn.Linear(hidden_size, hidden_size)
        self.silu = nn.SiLU()

        self.metric_w1 = nn.Parameter(torch.Tensor(1, hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, hidden_size))
        self.gating_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.gating_bias = nn.Parameter(torch.Tensor(1, hidden_size))
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.matmul(z1, z2.permute(0,2,1))

    def forward(self, input_tensor):
        # 1-level enhancement
        x_raw = input_tensor
        x_raw = x_raw * torch.sigmoid(x_raw.matmul(self.gating_weight)+self.gating_bias)
        x_m = torch.stack((self.metric_w1*x_raw, self.metric_w2*x_raw)).mean(0)
        item_sim = self.sim(x_m, x_m)
        item_sim[item_sim < 0] = 0.01
        
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        weights = F.softmax(self.weights, dim=0)
        attention_scores = (weights[0] * attention_scores + weights[1] * item_sim.unsqueeze(1)) / self.sqrt_attention_head_size

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        gate = self.silu(self.gate(context_layer))
        hidden_states = self.dense(context_layer * gate)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class MoE(nn.Module):
    def __init__(self, d_model, n_experts, expert_hidden_size, dropout):
        super(MoE, self).__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, expert_hidden_size),
            nn.GELU(),
            nn.Linear(expert_hidden_size, d_model)
        ) for _ in range(n_experts)])
        
        self.gate = nn.Linear(d_model, n_experts)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
    
    def forward(self, x):
        B, L, D = x.shape
        
        gate_scores = self.gate(x)
        gate_scores = F.softmax(gate_scores, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        
        hidden_states = torch.einsum('bldn,bln->bld', expert_outputs, gate_scores)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)
        
        return hidden_states