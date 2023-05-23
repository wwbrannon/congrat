'''
The dataset including both a graph object and the associated texts
'''

from typing import Callable, Dict, Any, Optional, List, Union

import copy
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as td

from torch_geometric.utils import to_networkx, to_dense_adj

import networkx as nx
from sklearn.decomposition import TruncatedSVD

import transformers as tf

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import (
    subgraph,
    remove_isolated_nodes,
    negative_sampling
)

from .. import utils as ut
from ..utils import _train_val_test_split


class BaseDataset(ABC, td.Dataset):
    def __init__(self,
        device: Union[torch.device, str] = 'cuda',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # torch device used for any heavy number crunching
        self.device = device

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def __collate__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def restrict(self,
        nodes: torch.Tensor,
        inplace: bool = True
    ) -> "BaseDataset":
        raise NotImplementedError()

    @property
    @abstractmethod
    def unique_node_ids(self):
        raise NotImplementedError()

    def split(self,
        split_proportions: Optional[List[float]] = [0.7, 0.1, 0.2],
        seed: Optional[int] = 42,
        splits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, "BaseDataset"]:
        if splits is None:
            splits = _train_val_test_split(
                self.unique_node_ids,
                split_proportions,
                seed
            )

        ret = {}
        for sp, nodes in splits.items():
            ret[sp] = self.restrict(nodes, inplace=False)

        return ret


class TextDataset(BaseDataset):
    def __init__(self,
        text: pd.DataFrame,

        tokenizer_name: str,
        mlm: bool = False,
        mlm_probability: float = 0.15,

        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.tokenizer_name = tokenizer_name
        self.tokenizer = tf.AutoTokenizer.from_pretrained(tokenizer_name)

        # some models like GPT-2 didn't have a padding token, so we need to
        # set one to do batching. it's fine that it wasn't trained this way,
        # we're using the attention mask and these tokens will be ignored
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer_params = {
            'return_tensors': 'pt',
            'return_attention_mask': True,
            'return_special_tokens_mask': True,

            'truncation': True,
            'padding': True,
            'max_length': self.tokenizer.model_max_length,
        }

        if mlm:
            self.enable_mlm(mlm_probability)
        else:
            self.disable_mlm()

        self.text_ids = torch.from_numpy(text['id'].to_numpy())
        self.text_node_ids = torch.from_numpy(text['node_id'].to_numpy())
        self.text = text['content'].to_numpy()

    def __len__(self) -> int:
        return len(self.text_node_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ret = {
            'text_node_ids': self.text_node_ids[idx],
            'text': self.text[idx]
        }

        return ret

    def tokenize(self, text: List[str]) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError('Must provide a tokenizer name to __init__')

        return self.tokenizer(text, **self.tokenizer_params)

    def enable_mlm(self, mlm_probability: float) -> "TextDataset":
        return self.configure_collation(True, mlm_probability)

    def disable_mlm(self) -> "TextDataset":
        return self.configure_collation(False, 0.0)

    def configure_collation(self,
        mlm: bool = False,
        mlm_probability: float = 0
    ) -> "TextDataset":
        self.mlm = mlm
        self.mlm_probability = mlm_probability

        self.collator = tf.DataCollatorForLanguageModeling(
            mlm = self.mlm,
            mlm_probability = self.mlm_probability,
            tokenizer = self.tokenizer,
            return_tensors = 'pt',
        )

        return self

    def collate_targets(self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        collated = self.collator([batch])

        for key in collated.keys():
            assert len(collated[key].shape) == 3
            collated[key] = collated[key].squeeze(0)

        if not self.mlm:
            collated['labels'] = F.pad(
                input = collated['labels'][:, 1:],
                pad = (0, 1),
                mode = 'constant',
                value = -100,
            )

        return collated

    def __collate__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {}

        with torch.no_grad():
            fields = samples[0].keys()
            for field in fields:
                vals = [s[field] for s in samples]

                if all(isinstance(s, torch.Tensor) for s in vals):
                    vals = torch.stack(vals, dim=0)

                ret[field] = vals

            encoded = self.tokenize(ret['text'])
            encoded = self.collate_targets(encoded)
            ret.update(encoded)

        return ret

    @property
    def unique_node_ids(self) -> torch.Tensor:
        return torch.unique(self.text_node_ids, dim=None, sorted=True)

    def restrict(self,
        nodes: torch.Tensor,
        inplace: bool = True
    ) -> "TextDataset":
        text_mask = torch.isin(self.text_node_ids, nodes)

        kwargs = {
            'text_ids': self.text_ids[text_mask],
            'text_node_ids': self.text_node_ids[text_mask],
            'text': self.text[text_mask.numpy()],
        }

        ret = self if inplace else copy.deepcopy(self)
        vars(ret).update(kwargs)

        return ret


class GraphDatasetMixin(BaseDataset):
    def __init__(self,
        graph_data: Data,
        drop_isolates: bool = True,

        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.graph_data = graph_data
        self._drop_isolates = drop_isolates

        if self._drop_isolates:
            self.drop_isolates(inplace=True)

        self.compute_neg_edge_index()

    def compute_mutuals(self,
        edge_index: torch.Tensor = None,
        num_nodes: int = None
    ) -> None:
        if edge_index is None:
            assert num_nodes is None

            edge_index = self.graph_data.edge_index
            num_nodes = self.graph_data.num_nodes

        A = to_dense_adj(edge_index).squeeze(0)
        if torch.cuda.is_available():
            A = A.to(self.device)  # verrry slow on CPU

        with torch.no_grad():
            mutual = A @ A.T  # mutual in/out edge counts
            mutual = F.normalize(mutual, p=2, dim=1)
            mutual = mutual @ mutual.T  # cosine similarity

        self.graph_data.sim_mutual = mutual.cpu()

    def compute_neg_edge_index(self,
        edge_index: torch.Tensor = None,
        num_nodes: int = None
    ) -> None:
        if edge_index is None:
            assert num_nodes is None

            edge_index = self.graph_data.edge_index
            num_nodes = self.graph_data.num_nodes

        neg_edges = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=self.graph_data.num_edges
        )

        self.graph_data.neg_edge_index = neg_edges

    @property
    def unique_node_ids(self) -> torch.Tensor:
        return torch.unique(self.graph_data.node_ids, dim=None, sorted=True)

    # overridable if subclasses need to do something else
    def _restrict_hook(self, targets: torch.Tensor) -> Dict[str, Any]:
        return {}

    # NOTE This is necessary and we can't just use random_split
    # because a) splitting is by node and b) the edge_index has
    # to be relabeled if the dataset is subset to only some nodes
    def restrict(self, targets: torch.Tensor, inplace: bool = False
                ) -> "GraphDataset":
        targets = torch.unique(targets, dim=None, sorted=True)

        #
        # First construct the new graph
        #

        graph_mask = (self.graph_data.node_ids[:, None] == targets).any(-1)

        new_node_ids = self.graph_data.node_ids[graph_mask]
        new_edge_index, _ = subgraph(graph_mask, self.graph_data.edge_index,
                                     relabel_nodes=True)

        graph_kwargs = {
            'edge_index': new_edge_index,
            'node_ids': new_node_ids
        }

        misc_keys = list(set(self.graph_data.keys) - set(graph_kwargs.keys()))
        for key in misc_keys:
            if key == 'neg_edge_index':
                continue

            obj = getattr(self.graph_data, key)

            if isinstance(obj, (torch.Tensor, SparseTensor)):
                graph_kwargs[key] = obj[graph_mask]
            elif isinstance(obj, list) and len(obj) == self.graph_data.num_nodes:
                indices = graph_mask.nonzero(as_tuple=False).view(-1).tolist()
                graph_kwargs[key] = [obj[i] for i in indices]
            else:
                graph_kwargs[key] = obj

        kwargs = dict(
            graph_data=Data(**graph_kwargs),
            **self._restrict_hook(targets)
        )

        ret = self if inplace else copy.deepcopy(self)
        vars(ret).update(kwargs)

        return ret

    def drop_isolates(self, inplace: bool = True) -> "GraphDataset":
        _, _, mask = remove_isolated_nodes(
            edge_index=self.graph_data.edge_index,
            edge_attr=None,

            # need this in case the isolates are the highest-numbered nodes;
            # without it, if there are no more nodes with an edge after them in
            # the numbering order, the inferred num_nodes will be the max of
            # edge_index and will be incorrect, leading to too short a mask
            num_nodes=self.graph_data.num_nodes
        )

        non_isolates = self.graph_data.node_ids[mask]

        return GraphDataset.restrict(self, non_isolates, inplace=inplace)

    def split(self,
        split_proportions: List[float] = [0.7, 0.1, 0.2],
        seed: Optional[int] = 42,
        splits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, "GraphDataset"]:
        splits = super().split(split_proportions, seed, splits)

        for name, ds in splits.items():
            if self._drop_isolates:
                ds.drop_isolates(inplace=True)

            ds.compute_neg_edge_index()

        return splits


class GraphDataset(GraphDatasetMixin):
    def __len__(self) -> int:
        return 1  # only the one graph

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ret = {}

        for key in self.graph_data.keys:
            ret['graph_' + key] = getattr(self.graph_data, key)

        return ret

    def __collate__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(samples) == 1

        return samples[0]

    def split(self,
        split_proportions: List[float] = [0.7, 0.1, 0.2],
        seed: Optional[int] = 42,
        splits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, "GraphDataset"]:
        splits = super().split(split_proportions, seed, splits)

        mn = splits['train'].graph_data.x.mean(dim=0)
        std = splits['train'].graph_data.x.std(dim=0)

        splits['train'].graph_data.x = splits['train'].graph_data.x - mn
        splits['train'].graph_data.x = splits['train'].graph_data.x / std

        splits['val'].graph_data.x = splits['val'].graph_data.x - mn
        splits['val'].graph_data.x = splits['val'].graph_data.x / std

        splits['test'].graph_data.x = splits['test'].graph_data.x - mn
        splits['test'].graph_data.x = splits['test'].graph_data.x / std

        return splits

class GraphTextDataset(GraphDatasetMixin, TextDataset):
    def __init__(self,
        random_data_debug: bool = False,
        max_texts_per_node: int = 0,

        transductive: bool = False,
        transductive_identity_features: bool = False,

        **kwargs: Any
    ) -> None:
        assert not (not transductive and transductive_identity_features), \
            "transductive_identity_features requires transductive = True"

        super().__init__(**kwargs)

        self.random_data_debug = random_data_debug
        self.max_texts_per_node = max_texts_per_node
        self.transductive = transductive
        self.transductive_identity_features = transductive_identity_features

        unique_text_node_ids = set(self.text_node_ids.tolist())
        unique_graph_node_ids = set(self.graph_data.node_ids.tolist())
        assert unique_text_node_ids <= unique_graph_node_ids

        if self.max_texts_per_node > 0:
            df = pd.concat([
                pd.Series(self.text_node_ids.numpy(), name='text_node_ids'),
                pd.Series(self.text, name='text')
            ], axis=1)

            def sampler(s):
                if s.shape[0] <= self.max_texts_per_node:
                    # df.sample raises if asked to sample more than exist
                    return s
                else:
                    return s.sample(n=self.max_texts_per_node, replace=False)

            df = df.groupby('text_node_ids').apply(sampler)

            self.text_node_ids = torch.from_numpy(df['text_node_ids'].to_numpy())
            self.text = df['text'].to_numpy()

        if self.transductive_identity_features:
            n = self.graph_data.x.shape[0]
            self.graph_data.x = torch.arange(n).unsqueeze(-1)

        if self.random_data_debug:
            self._i2w = { i : w for w, i in self.tokenizer.vocab.items() }
            self._i2w_keys = list(self._i2w.keys())

            special_ids = ['bos', 'eos', 'sep', 'cls', 'pad']
            special_ids = [s + '_token_id' for s in special_ids]
            special_ids = torch.tensor([
                getattr(self.tokenizer, s)
                for s in special_ids
                if hasattr(self.tokenizer, s)
            ])

            self._special_ids = special_ids

            # random graph input vectors
            self._random_x = torch.randn(*self.graph_data.x.shape)

            # random graph edges - note *different* randperms
            ei = self.graph_data.edge_index
            self._random_edge_index = torch.stack([
                ei[0, torch.randperm(ei.shape[0])],
                ei[1, torch.randperm(ei.shape[0])],
            ], dim=0)

    def split(self,
        split_proportions: List[float] = [0.7, 0.1, 0.2],
        seed: Optional[int] = 42,
        splits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, "GraphDataset"]:
        splits = super().split(split_proportions, seed, splits)

        for name, ds in splits.items():
            if self.transductive and hasattr(self.graph_data, 'tsvd'):
                # don't want leakage from the other splits
                # NOTE that putting this here is kind of a hack, shoud probably
                # be factored out somewhere else
                G = to_networkx(ds.graph_data)
                nodelist = ds.graph_data.node_ids.numpy()
                nx.relabel_nodes(G, pd.Series(nodelist).to_dict(), copy=False)

                adj = nx.adjacency_matrix(G, nodelist=nodelist).astype(float)
                tsvd = TruncatedSVD(64, algorithm='arpack').fit_transform(adj)

                # NOTE hack put ds.graph_data.x here to have this data be used
                # as the graph node features
                ds.graph_data.tsvd = torch.from_numpy(tsvd)

            if self.transductive_identity_features:
                n = ds.graph_data.x.shape[0]
                ds.graph_data.x = torch.arange(n).unsqueeze(-1)
            else:
                # NOTE having this here is also kind of a hack, should probably
                # be factored differently
                mn = splits['train'].graph_data.x.mean(dim=0)
                std = splits['train'].graph_data.x.std(dim=0)

                splits['train'].graph_data.x = splits['train'].graph_data.x - mn
                splits['train'].graph_data.x = splits['train'].graph_data.x / std

                splits['val'].graph_data.x = splits['val'].graph_data.x - mn
                splits['val'].graph_data.x = splits['val'].graph_data.x / std

                splits['test'].graph_data.x = splits['test'].graph_data.x - mn
                splits['test'].graph_data.x = splits['test'].graph_data.x / std

        return splits

    def _restrict_hook(self, targets: torch.Tensor) -> Dict[str, Any]:
        text_mask = torch.isin(self.text_node_ids, targets)

        return {
            'text_ids': self.text_ids[text_mask],
            'text_node_ids': self.text_node_ids[text_mask],
            'text': self.text[text_mask.numpy()],
        }

    def restrict(self, targets: torch.Tensor, inplace: bool = False
                ) -> "GraphTextDataset":
        if self.transductive:
            # note we need to explicitly pass self
            return TextDataset.restrict(self, targets, inplace)
        else:
            return GraphDataset.restrict(self, targets, inplace)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ret = super().__getitem__(idx)

        for key in self.graph_data.keys:  # copy references, not values
            ret['graph_' + key] = getattr(self.graph_data, key)

        return ret

    def __collate__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
        Collate multiple samples into a batch.

        Each sample is assumed to be a dict. Its keys which are torch.tensors
        are stacked together along a new dim 0, while other keys are grouped
        together in lists. If keys starting with 'graph_' are present, it is
        duplicated and and stacked in the collated batch; rather the batch has
        only the value from the first sample. (This is because this representation
        of the graph is shared across examples and we don't want to waste memory
        copying it many times.)
        '''

        if len(samples) == 0:
            return {}

        # these are shared across the data points (see __getitem__), and we
        # don't want to waste (a lot of) memory collating them for every
        # example
        graph_vals = {
            k : v
            for k, v in samples[0].items()
            if k.startswith('graph_')
        }

        samples = [
            { k : v for k, v in s.items() if not k.startswith('graph_')}
            for s in samples
        ]

        ret = super().__collate__(samples)
        ret = dict(ret, **graph_vals)

        if self.random_data_debug:
            ret['graph_x'] = self._random_x
            ret['graph_edge_index'] = self._random_edge_index

            inds = torch.randint(len(self._i2w), (ret['input_ids'].numel(),))
            tmp = [self._i2w_keys[i] for i in inds]
            tmp = torch.tensor(tmp).view(*ret['input_ids'].shape)

            special_mask = torch.isin(ret['input_ids'], self._special_ids)
            tmp[special_mask] = ret['input_ids'][special_mask]
            ret['input_ids'] = tmp

            ret['text'] = [
                self.tokenizer.decode(ret['input_ids'][i, :])
                for i in range(ret['input_ids'].shape[0])
            ]

        return ret


# The loss we want to use this with is batch-wise contrastive (every element of
# the batch is compared against every other element of the batch), so a) we
# want to use large batches and b) it's not totally clear what an epoch is,
# because how examples are batched influences what's learned from seeing those
# examples, so we do the batching in the dataset rather than the dataloader and
# just allow specifying how many batches to draw.
class BatchGraphTextDataset(BaseDataset):
    def __init__(self,
        dataset: GraphTextDataset,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        assert batch_size > 0

        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # This is an ndarray of strings, can't be represented in torch.
        # We also don't want to compute it on every fetch of a batch, it'd be
        # much too slow compared to doing it once here
        text_unique_ids = np.unique(self.dataset.text, return_inverse=True)[1]
        text_unique_ids = torch.from_numpy(text_unique_ids)
        self.text_unique_ids = text_unique_ids

        self.refresh_dataset_indices()

    def next_epoch(self) -> None:
        self.epoch += 1

        self.refresh_dataset_indices()

    # note that just using randperm to shuffle the indices results in a really
    # large rate of within-batch dupes. this results in many fewer, but still
    # doesn't avoid it entirely.
    def refresh_dataset_indices(self) -> None:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        node_inds = [
            torch.where(self.dataset.text_node_ids == node)[0]
            for node in self.dataset.text_node_ids.unique()
        ]

        max_length = max([n.shape[0] for n in node_inds])

        node_inds = torch.stack([
            F.pad(n, (0, max_length - n.shape[0]), mode='constant', value=-1)
            for n in node_inds
        ], dim=0)

        # shuffle nodes
        row_perm = torch.randperm(node_inds.shape[0], generator=g)
        node_inds = node_inds[row_perm, :]

        # shuffle texts within each node
        col_perm = torch.randperm(node_inds.shape[1], generator=g)
        node_inds = node_inds[:, col_perm]

        node_inds = node_inds.T.flatten()  # the .T interleaves nodes
        node_inds = node_inds[node_inds != -1]

        self.dataset_indices = node_inds

    def _indices_for_batch(self,
        batch: int,
        pad_to_same_length: bool = False
    ) -> List[int]:
        assert batch >= 0

        start_idx = batch * self.batch_size

        if pad_to_same_length:
            # We have multiple texts per node and need batches which are unique
            # on both nodes and texts. It's really surprisingly hard to do this
            # during computation of the dataset indices above - any obvious way
            # to do it is infeasibly slow - so instead we're going to randomly
            # shuffle them and dedupe each batch. To avoid different batches
            # thereby having different sizes, we're actually going to take each
            # batch and the next batch, dedupe preferring the earlier
            # occurrences as argunique does, and then return the first
            # self.batch_size indices. In general this means that we'll pad out
            # each batch with a (randomly chosen, because the indices are a
            # randperm) couple elements of the next batch and each epoch will
            # go over a small fraction of datapoints twice. This is still a
            # better solution than the alternatives, given how hard it is to
            # construct exactly the right set of batch indices.
            end_idx = int((batch + 1.5) * self.batch_size)
            if end_idx >= self.dataset_indices.shape[0]:
                # avoid crashing out on the last batch
                end_idx = (batch + 1) * self.batch_size
        else:
            end_idx = (batch + 1) * self.batch_size

        indices = self.dataset_indices[start_idx:end_idx]

        # dedupe by text
        uniques = self.text_unique_ids[indices]
        indices = indices[ut.argunique(uniques), ...]

        # dedupe by node ID
        node_ids = self.dataset.text_node_ids[indices]
        indices = indices[ut.argunique(node_ids), ...]

        return indices[:self.batch_size]  # selects up to this many

    def __len__(self) -> int:
        num_batches = len(self.dataset) / self.batch_size
        num_batches = torch.ceil(torch.tensor(num_batches)).long().item()

        return num_batches

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self._indices_for_batch(idx)

        batch = [self.dataset[idx.item()] for idx in indices]
        batch = self.dataset.__collate__(batch)

        return batch

    def __collate__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(samples) == 1

        return samples[0]

    @property
    def unique_node_ids(self):
        return self.dataset.unique_node_ids

    def restrict(self,
        nodes: torch.Tensor,
        inplace: bool = True
    ) -> "BaseDataset":
        raise NotImplementedError()

    def split(self,
        split_proportions: List[float] = [0.7, 0.1, 0.2],
        seed: Optional[int] = 42,
        splits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, "TextDataset"]:
        raise NotImplementedError()
