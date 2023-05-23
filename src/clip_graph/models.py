from typing import Optional, Dict, Any, Callable, Union

import io
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

import transformers as tf

from . import utils as ut

#
# Base classes
#


class ModelBase(nn.Module):
    '''
    Base class for models.
    '''

    pass


#
# Graph models
#

class GATLayer(ModelBase):
    def __init__(self, in_channels: int = 64,
                 out_channels: Optional[int] = None,
                 num_heads: int = 4, dropout: float = 0.1,
                 d_feedforward: int = 128) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_feedforward = d_feedforward

        # TODO maybe use GATv2Conv?
        self.conv = gnn.GATConv(in_channels, out_channels, heads=num_heads,
                                dropout=dropout, concat=True)
        self.linear1 = nn.Linear(out_channels * num_heads, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, out_channels)

        self.out_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor
               ) -> torch.Tensor:
        x = self.conv(x, edge_index)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = self.out_norm(x)

        return x


class GATMod(ModelBase):
    def __init__(self,
        in_channels: Optional[int] = None,
        num_nodes: Optional[int] = None,

        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        d_feedforward: int = 128,

        out_channels: Optional[int] = None,
    ) -> None:
        assert num_layers >= 1
        assert in_channels is not None or num_nodes is not None
        assert in_channels is None or num_nodes is None

        super().__init__()

        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_feedforward = d_feedforward
        self.out_channels = out_channels

        self.dropout = nn.Dropout(dropout)

        if self.in_channels is not None:
            self.embed = nn.Linear(self.in_channels, self.hidden_channels)
        else:  # self.num_nodes is not None
            self.embed = nn.Embedding(self.num_nodes, self.hidden_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers += [
                GATLayer(hidden_channels, hidden_channels, num_heads,
                         dropout=dropout, d_feedforward=d_feedforward)
            ]

        if self.out_channels is not None:
            self.head = nn.Sequential(
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
                nn.Linear(self.hidden_channels, self.out_channels)
            )
        else:
            self.head = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor
               ) -> torch.Tensor:
        if self.num_nodes is not None:  # these are node IDs
            x = x.squeeze(-1)
        assert len(x.shape) == 2 if self.in_channels is not None else 1

        # dropout as early as possible: on the input data if we can, otherwise
        # on the embeddings
        if self.in_channels is not None:
            x = self.dropout(x)
            x = self.embed(x) * (self.hidden_channels ** 0.5)
        else:
            x = self.embed(x) * (self.hidden_channels ** 0.5)
            x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = x + layer(x, edge_index)

        ret = {'last_hidden_state': x}
        if self.head is not None:
            ret['output'] = self.head(x)
        else:
            ret['output'] = None

        return ret


#
# Language models
#


class LMForSentenceEmbedding(ModelBase):
    def __init__(self,
        model: Union[str, torch.nn.Module],
        pooling_mode: str = 'cls',
        normalize: bool = True
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            self.model = tf.AutoModel.from_pretrained(model)

            # we don't use this and it causes unused-parameters errors on DDP
            if hasattr(self.model, 'pooler') and self.model.pooler is not None:
                self.model.pooler = None
        else:
            self.model = model

        self.pooling_mode = pooling_mode
        self.normalize = normalize

    def _get_pooler(self, pooling_mode: str) -> Callable:
        if self.pooling_mode == 'mean':
            return self.mean_pool
        elif self.pooling_mode == 'cls':
            return self.cls_pool
        elif self.pooling_mode == 'max':
            return self.max_pool
        else:
            raise ValueError('Unsupported pooling mode')

    # NOTE all three of these poolers pool over the second (sequence) dimension
    @staticmethod
    def mean_pool(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(embeds.size())

        # mask.sum(dim=1) has the same dimension as sum(embeds * mask, dim=1)
        # so the division is elementwise, no broadcasting, no opportunity for
        # e.g. unexpected batch size to screw it up
        ret = torch.sum(embeds * mask, dim=1)
        ret /= torch.clamp(mask.sum(dim=1), min=1e-9)

        return ret

    @staticmethod
    def cls_pool(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return embeds[:, 0, ...]

    @staticmethod
    def max_pool(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(embeds.size())

        if copy:
            embeds = torch.clone(embeds)

        embeds[mask == 0] = -1e9  # never choose a padding token

        return torch.max(embeds, 1)[0]

    def sentence_embeddings(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: Optional[str] = None,
        normalize: Optional[bool] = None
    ) -> torch.Tensor:
        pooling_mode = ut.coalesce(pooling_mode, self.pooling_mode)
        normalize = ut.coalesce(normalize, self.normalize)

        embeds = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )['last_hidden_state']

        pooler = self._get_pooler(pooling_mode)
        embeds = pooler(embeds, attention_mask)

        if normalize:
            embeds = F.normalize(embeds, p=2, dim=1)

        return embeds

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.sentence_embeddings(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

    @property
    def config(self) -> Union[tf.PretrainedConfig, Any]:
        return self.model.config


class LMForPretrain(ModelBase):
    def __init__(self,
        model: Union[str, torch.nn.Module],
        mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert mode is None or mode in ('masked', 'causal')
        self.mode = mode

        if isinstance(model, str) and self.mode == 'masked':
            self.model = tf.AutoModelForMaskedLM.from_pretrained(model)
        elif isinstance(model, str):  # self.mode == 'causal'
            self.model = tf.AutoModelForCausalLM.from_pretrained(model)
        else:
            self.model = model

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )['logits']

    def to_sentence_embedding(self, **kwargs: Any) -> LMForSentenceEmbedding:
        return LMForSentenceEmbedding(self.model.base_model, **kwargs)

    @property
    def config(self) -> Union[tf.PretrainedConfig, Any]:
        return self.model.config


#
# CLIP-graph model
#


class ProjectHead(ModelBase):
    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim

        self.linear1 = nn.Linear(input_dim, target_dim)
        self.linear2 = nn.Linear(target_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = self.norm(x)

        return x


class ClipGraph(ModelBase):
    def __init__(self,
        lm: ModelBase,
        gnn: ModelBase,

        embed_dim: int = 768,
        dropout: float = 0,
        tau_init: float = 2.7,
        max_tau: Optional[float] = None,

        debug_checks: bool = False
    ) -> None:
        super().__init__()

        self.lm = lm
        self.gnn = gnn

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.tau_init = tau_init
        self.max_tau = max_tau

        self.debug_checks = debug_checks

        self.lm_proj = ProjectHead(
            self.lm.config.hidden_size,
            self.embed_dim,
        )

        if self.gnn.out_channels is not None:
            gnn_out_channels = self.gnn.out_channels
        else:
            gnn_out_channels = self.gnn.hidden_channels

        self.gnn_proj = ProjectHead(
            gnn_out_channels,
            self.embed_dim,
        )

        self.register_parameter(
            name='tau',
            param=nn.Parameter(torch.tensor(self.tau_init, requires_grad=True))
        )

    def embed_text(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.lm(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        if self.debug_checks:
            assert len(embeds.shape) == 2
            assert not torch.isinf(embeds).any().item()
            assert not torch.isnan(embeds).any().item()

        embeds = self.lm_proj(embeds)

        return embeds

    def embed_nodes(self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.gnn(x, edge_index)

        if embeds['output'] is not None:
            embeds = embeds['output']
        else:
            embeds = embeds['last_hidden_state']

        if self.debug_checks:
            assert len(embeds.shape) == 2
            assert not torch.isinf(embeds).any().item()
            assert not torch.isnan(embeds).any().item()

        embeds = self.gnn_proj(embeds)

        return embeds

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                graph_x: torch.Tensor, graph_edge_index: torch.Tensor,
                node_index: torch.Tensor) -> torch.Tensor:
        if self.debug_checks:
            assert not torch.isinf(input_ids).any().item()
            assert not torch.isnan(input_ids).any().item()

            assert not torch.isinf(attention_mask).any().item()
            assert not torch.isnan(attention_mask).any().item()

            assert not torch.isinf(graph_x).any().item()
            assert not torch.isnan(graph_x).any().item()

            assert not torch.isinf(graph_edge_index).any().item()
            assert not torch.isnan(graph_edge_index).any().item()

            assert not torch.isinf(node_index).any().item()
            assert not torch.isnan(node_index).any().item()

        lm_embeds = self.embed_text(input_ids, attention_mask)
        lm_embeds = F.normalize(lm_embeds, p=2, dim=1)
        lm_embeds = self.dropout(lm_embeds)

        if self.debug_checks:
            assert not torch.isinf(lm_embeds).any().item()
            assert not torch.isnan(lm_embeds).any().item()

        gnn_embeds = self.embed_nodes(graph_x, graph_edge_index)
        gnn_embeds = F.normalize(gnn_embeds, p=2, dim=1)
        gnn_embeds = gnn_embeds[node_index, ...]
        gnn_embeds = self.dropout(gnn_embeds)

        if self.debug_checks:
            assert not torch.isinf(gnn_embeds).any().item()
            assert not torch.isnan(gnn_embeds).any().item()

        temperature = self.tau.exp()
        if self.max_tau is not None:
            max_temp = torch.tensor(self.max_tau).exp().item()
            temperature = temperature.clamp(-max_temp, max_temp)

        ret = lm_embeds @ gnn_embeds.T * temperature

        if self.debug_checks:
            assert not torch.isinf(ret).any().item()
            assert not torch.isnan(ret).any().item()

        return ret
