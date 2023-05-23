from typing import Dict, List, Tuple, Any, Callable, Optional, Union

import os
import io
import functools as ft
import collections as cl
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers as tf

from torch_geometric import nn as gnn

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl

from . import losses as ls
from . import optimizers as op
from . import models as md
from . import utils as ut


#
# Base class
#


@torch.no_grad()
def set_dropout(module, p=0):  # NOTE this is an awful hack
    names = ['dropout', 'p_dropout', 'dropout_p', 'prob_dropout',
             'dropout_prob']

    for name in names:
        if hasattr(module, name):
            att = getattr(module, name)

            if isinstance(att, float):
                setattr(module, name, p)
            elif isinstance(att, nn.Dropout):
                att.p = p
            else:
                raise ValueError("Module has unsupported configuration")


@torch.no_grad()
def set_layernorm_eps(module, eps=1e-4):  # this is also an awful hack
    if isinstance(module, nn.LayerNorm):
        module.eps = eps


class LitBase(ABC, pl.LightningModule):
    '''
    Base class for LightningModules
    '''

    def __init__(self,
        model: Optional[md.ModelBase] = None,
        model_class_name: Optional[str] = None,
        model_file: Optional[Union[io.IOBase, str]] = None,
        model_params: Optional[Dict[str, Any]] = None,

        lr: float = 3e-5,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
        betas: Union[Tuple[float, float], List[float]] = (0.9, 0.999),
    ) -> None:
        super().__init__()

        if model_class_name is None and model_params is not None:
            raise ValueError('Extra keyword arguments are only meaningful '
                             'together with model_class_name, when they are '
                             'passed to its __init__ method')

        if ut.count(model, model_class_name, model_file) != 1:
            raise ValueError('Must provide exactly one of model, '
                             'model_class_name, model_file')

        if model is not None:
            self.model = model
        elif model_file is not None:
            if isinstance(model_file, str):
                with open(model_file, 'rb') as f:
                    self.model = torch.load(f)
            elif isinstance(model_file, io.IOBase):
                self.model = torch.load(model_file)
            else:
                raise ValueError('model_file must be str or file-like')
        else:  # model_class_name is not None
            model_params = {} if model_params is None else model_params
            self.model = getattr(md, model_class_name)(**model_params)

        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = tuple(betas)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @abstractmethod
    def step(self,
        batch: Dict[str, Any],
        batch_idx: int,
        split: str
    ) -> torch.Tensor:
        raise NotImplementedError()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.step(*args, **kwargs, split='train')

    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.step(*args, **kwargs, split='val')

    def test_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.step(*args, **kwargs, split='test')

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )


#
# Graph models
#


LitGAEOutput = cl.namedtuple('LitGAEOutput', ['z'])


class LitGAE(LitBase):
    # wrap the input model for pretraining
    autoencoder_class = gnn.GAE

    def __init__(self,
        node_features_keys: List[str] = ['graph_x'],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = self.autoencoder_class(encoder=self.model)
        self.node_features_keys = node_features_keys

        # Example = cl.namedtuple('Example', ['x', 'edge_index'])
        # self.example_input_array = Example(
        #     # 200 (fake) nodes, 768 node data dims, 1000 edges
        #     torch.randn(200, 768),
        #     torch.randint(0, 200, (2, 1000)),
        # )

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        out = self.model.encoder(x, edge_index)

        if isinstance(out, torch.Tensor):
            pass
        elif isinstance(out, cl.abc.Mapping) and out['output'] is not None:
            out = out['output']
        elif isinstance(out, cl.abc.Mapping):
            out = out['last_hidden_state']
        else:
            raise ValueError('Bad output from encoder: ' + str(out))

        return LitGAEOutput(out)

    def loss(self,
        batch: Dict[str, Any],
        outputs: LitGAEOutput,
    ) -> torch.Tensor:
        return self.model.recon_loss(
            outputs.z,
            batch['graph_edge_index'],
            batch['graph_neg_edge_index'],
        )

    def step(self,
        batch: Dict[str, Any],
        batch_idx: int,
        split: str
    ) -> torch.Tensor:
        x = torch.cat([batch[k] for k in self.node_features_keys], dim=1)
        x = x.float()

        outputs = self(x, batch['graph_edge_index'])
        loss = self.loss(batch, outputs)

        auc, ap = self.model.test(
            outputs.z,
            batch['graph_edge_index'],
            batch['graph_neg_edge_index'],
        )

        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=1,
                 rank_zero_only=True)

        self.log(f'{split}_auc', auc, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=1,
                 rank_zero_only=True)

        self.log(f'{split}_ap', ap, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=1,
                 rank_zero_only=True)

        return loss


class LitClipGraphGNNTrain(LitGAE):
    def __init__(self, ckpt_path: str, **kwargs: Any) -> None:
        assert 'model' not in kwargs.keys() or kwargs['model'] is None

        mod = LitClipGraph.load_from_checkpoint(ckpt_path).model.gnn
        kwargs['model'] = mod

        super().__init__(**kwargs)


LitVGAEOutput = cl.namedtuple('LitGAEOutput', ['z', 'mu', 'logstd'])


class LitVGAE(LitGAE):
    autoencoder_class = gnn.VGAE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # need to divide these into mu and logstd
        assert self.model.encoder.out_channels % 2 == 0

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        encoded = super().forward(x, edge_index).z

        latent_dim = int(self.model.encoder.out_channels / 2)
        mu, logstd = encoded[:, :latent_dim], encoded[:, latent_dim:]
        logstd = logstd.clamp(max=10)

        return LitVGAEOutput(self.model.reparametrize(mu, logstd), mu, logstd)

    def loss(self,
        batch: Dict[str, Any],
        outputs: LitVGAEOutput,
    ) -> torch.Tensor:
        recon_loss = super().loss(batch, outputs)

        B = outputs.z.shape[0]
        variational_loss = 1/B * self.model.kl_loss(
            outputs.mu,
            outputs.logstd
        )

        return recon_loss + variational_loss


#
# Language models
#


class LitPretrainLM(LitBase):
    def __init__(self,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,

        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if (num_warmup_steps is None and num_training_steps is not None) or \
           (num_training_steps is None and num_warmup_steps is not None):
            raise ValueError('Must provide both or none of num_training_steps'
                             'and num_warmup_steps')

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        # Example = cl.namedtuple('Example', ['input_ids', 'attention_mask'])
        # self.example_input_array = Example(
        #     # [
        #     #     'This is a sentence.',
        #     #     'Look, here is another sentence.',
        #     #     'And finally here is a third sentence.',
        #     # ],

        #     torch.LongTensor([
        #         [   0, 2027, 2007, 1041, 6255, 1016,    2,    1,    1,    1],
        #         [   0, 2302, 1014, 2186, 2007, 2182, 6255, 1016,    2,    1],
        #         [   0, 2002, 2637, 2186, 2007, 1041, 2357, 6255, 1016,    2]
        #     ]),

        #     torch.LongTensor([
        #         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #     ]),
        # )

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def step(self,
        batch: Dict[str, Any],
        batch_idx: int,
        split: str
    ) -> torch.Tensor:
        logits = self(batch['input_ids'], batch['attention_mask'])
        B, W, H = logits.shape

        loss = self.criterion(
            logits.reshape(B * W, H),
            batch['labels'].reshape(B * W),
        )

        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True, batch_size=B,
                 rank_zero_only=True)

        return loss

    @property
    def criterion(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    def configure_optimizers(self) -> Optimizer:
        opt = super().configure_optimizers()

        if self.num_training_steps is None:
            return opt
        else:
            fn = ft.partial(  # ft.partial objects are picklable for >1 GPU
                op.warmup_lambda,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps
            )

            sched = torch.optim.lr_scheduler.LambdaLR(opt, fn)

            lr_dict = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1
            }

            return {
                'optimizer': opt,
                'lr_scheduler': lr_dict
            }


class LitClipGraphLMTrain(LitPretrainLM):
    def __init__(self,
        ckpt_path: Optional[str] = None,
        restart_ckpt_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        assert ckpt_path is None or restart_ckpt_path is None
        assert ckpt_path is not None or restart_ckpt_path is not None
        assert 'model' not in kwargs.keys() or kwargs['model'] is None

        if ckpt_path is not None:
            mod = LitClipGraph.load_from_checkpoint(ckpt_path).model.lm.model
            mod2 = tf.AutoModelForCausalLM.from_config(mod.config)
            mod2.transformer.load_state_dict(mod.state_dict())
            kwargs['model'] = mod2
        else:  # restart_ckpt_path is not None:
            mod = LitClipGraphLMTrain.load_from_checkpoint(restart_ckpt_path).model
            kwargs['model'] = mod

        super().__init__(**kwargs)

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

#
# CLIP-graph model
#


class LitClipGraph(LitBase):
    def __init__(self,
        lm_name: Optional[str] = None,
        lm_ckpt_path: Optional[str] = None,
        lm_pooling_mode: str = 'mean',
        lm_normalize: bool = False,

        gnn_class_name: Optional[str] = None,
        gnn_params: Optional[Dict[str, Any]] = None,
        gnn_ckpt_path: Optional[str] = None,
        gnn_ckpt_dropout: Optional[float] = None,
        gnn_node_features_keys: List[str] = ['graph_x'],

        sim_smoothing: float = 0.0,
        sim_weights: Optional[str] = None,

        embed_dim: int = 768,
        dropout: float = 0.0,
        tau_init: float = 0.07,
        max_tau: Optional[float] = None,
        lm_layernorm_eps: Optional[float] = None,
        cycle_length_steps: Optional[int] = None,

        debug_checks: bool = False,

        **kwargs: Any
    ) -> None:
        if ut.count(lm_name, lm_ckpt_path) != 1:
            raise ValueError('Must provide exactly one of lm_name and '
                             'lm_ckpt_file')

        if ut.count(gnn_class_name, gnn_ckpt_path) != 1:
            raise ValueError('Must provide exactly one of gnn_class_name and '
                             'gnn_ckpt_file')

        if gnn_params is not None and gnn_class_name is None:
            raise ValueError('Can only specify gnn_params with gnn_class_name')

        if lm_ckpt_path is not None:
            lm = LitPretrainLM.load_from_checkpoint(lm_ckpt_path) \
                       .model \
                       .to_sentence_embedding(
                           pooling_mode=lm_pooling_mode,
                           normalize=lm_normalize
                       )
        else:
            lm = md.LMForSentenceEmbedding(
                model=lm_name,
                pooling_mode=lm_pooling_mode,
                normalize=lm_normalize
            )

        if gnn_ckpt_path is not None:
            gnn = LitGraphModel.load_from_checkpoint(gnn_ckpt_path)

            # throw away the LightningModule and the (V)GAE - we just want the
            # node embeddings
            gnn = gnn.model.encoder.model

            if gnn_ckpt_dropout is not None:
                gnn.apply(lambda m: set_dropout(m, p=gnn_ckpt_dropout))
        else:
            gnn_params = {} if gnn_params is None else gnn_params
            gnn = getattr(md, gnn_class_name)(**gnn_params)

        kwargs['model'] = md.ClipGraph(
            lm = lm,
            gnn = gnn,
            embed_dim = embed_dim,
            dropout = dropout,
            tau_init = tau_init,
            max_tau = max_tau,

            debug_checks=debug_checks
        )

        super().__init__(**kwargs)

        self.gnn_node_features_keys = gnn_node_features_keys
        self.lm_pooling_mode = lm_pooling_mode
        self.lm_normalize = lm_normalize
        self.sim_smoothing = sim_smoothing
        self.sim_weights = sim_weights
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.tau_init = tau_init
        self.lm_layernorm_eps = lm_layernorm_eps
        self.cycle_length_steps = cycle_length_steps
        self.debug_checks = debug_checks

        sim_weight_options = ['identity', 'exp', 'exp_thick_tail']
        if self.sim_weights is not None and self.sim_weights not in sim_weight_options:
            raise ValueError(f'Invalid sim_weights option: {self.sim_weights}')

        if self.lm_layernorm_eps:
            fn = ft.partial(set_layernorm_eps, eps=self.lm_layernorm_eps)
            self.model.lm.apply(fn)

        # Example = cl.namedtuple('Example', ['input_ids', 'attention_mask',
        #                                     'graph_x', 'graph_edge_index',
        #                                     'graph_node_ids', 'text_node_ids'])
        # self.example_input_array = Example(
        #     # [
        #     #     'This is a sentence.',
        #     #     'Look, here is another sentence.',
        #     #     'And finally here is a third sentence.',
        #     # ],

        #     torch.LongTensor([
        #         [   0, 2027, 2007, 1041, 6255, 1016,    2,    1,    1,    1],
        #         [   0, 2302, 1014, 2186, 2007, 2182, 6255, 1016,    2,    1],
        #         [   0, 2002, 2637, 2186, 2007, 1041, 2357, 6255, 1016,    2]
        #     ]),

        #     torch.LongTensor([
        #         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #     ]),

        #     # 200 (fake) nodes, 768 node data dims, 1000 edges
        #     torch.randn(200, 768) if self.model.gnn.in_channels is not None \
        #     else torch.randint(0, 200, (200,)),

        #     torch.randint(0, 200, (2, 1000)),

        #     # (fake) node IDs which are just sequential ints, and 3 randomly
        #     # chosen ones for the texts to have come from
        #     torch.arange(0, 200),
        #     torch.randint(0, 200, (3,)),
        # )

        self.save_hyperparameters(ignore=['model'])

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_node_ids: torch.Tensor,
        text_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        node_index = (graph_node_ids == text_node_ids[:, None])
        node_index = node_index.nonzero()[:, 1]

        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            graph_x = graph_x,
            graph_edge_index = graph_edge_index,
            node_index = node_index
        )

    def step(self,
        batch: Dict[str, Any],
        batch_idx: int,
        split: str
    ) -> torch.Tensor:
        if self.debug_checks:
            # Sanity check that there aren't duplicates
            n_nodes = torch.unique(batch['graph_node_ids']).shape[0]
            n_texts = len(set(batch['text']))

            assert batch['graph_node_ids'].shape[0] == n_nodes
            assert len(batch['text']) == n_texts

        graph_x = torch.cat([
            batch[k]
            for k in self.gnn_node_features_keys
        ], dim=1)

        # NOTE sometimes you need a .half() here under mixed precision for
        # reasons I don't understand - data issue?
        # graph_x = graph_x.float()  # or .half()

        logits = self(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            graph_x = graph_x,
            graph_edge_index = batch['graph_edge_index'],
            graph_node_ids = batch['graph_node_ids'],
            text_node_ids = batch['text_node_ids'],
        )

        if self.sim_smoothing > 0:
            inds = (batch['text_node_ids'][:, None] == batch['graph_node_ids'])
            inds = inds.nonzero()[:, 1]

            sims = batch['graph_sim_mutual'][inds, ...][..., inds]
            sims = torch.clone(sims.detach())

            # distribution over *other* nodes; this is equivalent to
            # sims.fill_diagonal_(0) but is out-of-place so deepspeed doesn't
            # barf when we try to train
            sims = sims - torch.diagflat(torch.diagonal(sims))

            loss = self.criterion(
                logits = logits,
                sims = sims,
                sim_weights = self.sim_weights,
                alpha = self.sim_smoothing
            )
        else:
            loss = self.criterion(logits)

        if self.debug_checks:
            assert not torch.isinf(loss).any().item()
            assert not torch.isnan(loss).any().item()

        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True,
                 logger=True, prog_bar=True, sync_dist=True,
                 batch_size=batch['text_node_ids'].shape[0],
                 rank_zero_only=True)

        if split == 'train':
            self.log(f'{split}_tau', self.model.tau, on_step=True,
                    on_epoch=True, logger=True, sync_dist=True,
                    batch_size=batch['text_node_ids'].shape[0],
                    rank_zero_only=True)

        return loss

    @property
    def criterion(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return ls.square_contrastive_loss

    def configure_optimizers(self) -> Optimizer:
        opt = super().configure_optimizers()

        if self.cycle_length_steps is None:
            return opt
        else:
            sched = CosineAnnealingWarmRestarts(opt, self.cycle_length_steps)

            lr_dict = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1
            }

            return {
                'optimizer': opt,
                'lr_scheduler': lr_dict
            }
