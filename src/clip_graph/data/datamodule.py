'''
The pytorch-lightning LightningDataModules encapsulating our datasets and their
splits, loaders, etc.
'''

from typing import List, Optional, Any, Union

import os
import pickle
import collections as cl
import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.utils.data as td
import transformers as tf

import sentence_transformers as st

import pytorch_lightning as pl

from sklearn.decomposition import TruncatedSVD

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_dense_adj, to_undirected

from .dataset import (
    BaseDataset,
    GraphDataset, TextDataset, GraphTextDataset,
    BatchGraphTextDataset
)

from ..utils import _train_val_test_split

#
# Base data module functionality
#


class BaseDataModule(ABC, pl.LightningDataModule):
    def __init__(self,
        data_dir: str,
        batch_size: int = 32,
        split_proportions: List[float] = [0.7, 0.1, 0.2],
        seed: int = 42,
        pin_memory: bool = True,
        num_workers: int = 0,
        device: Union[torch.device, str] = 'cuda',

        **kwargs: Any
    ) -> None:
        if isinstance(split_proportions, (list, tuple)):
            split_proportions = torch.tensor(split_proportions)
        assert torch.isclose(split_proportions.sum(), torch.tensor(1.0))

        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_proportions = split_proportions  # [train, val, test]
        self.seed = seed

        self.pin_memory = pin_memory
        self.num_workers = mp.cpu_count() if num_workers == -1 else num_workers
        self.device = device

        self._edgelist_path = os.path.join(self.data_dir, 'raw', 'graph.csv')
        self._text_path = os.path.join(self.data_dir, 'raw', 'texts.csv')
        self._node_data_path = os.path.join(self.data_dir, 'raw', 'node-data.csv')
        self._derived_path = os.path.join(self.data_dir, 'derived', 'node-properties.csv')

        self._split_nodes = None

    @abstractmethod
    def prepare_data(self) -> "BaseDataModule":
        raise NotImplementedError()

    @abstractmethod
    def setup(self) -> "BaseDataModule":
        raise NotImplementedError()

    def post_compute_split_nodes_hook(self) -> "BaseDataModule":
        pass

    def post_split_hook(self) -> "BaseDataModule":
        pass

    def _get_edgelist(self) -> pd.DataFrame:
        return pd.read_csv(self._edgelist_path, sep='\t')

    def _get_text(self) -> pd.DataFrame:
        ret = pd.read_csv(self._text_path, sep='\t')

        if 'id' not in ret.columns:
            ret['id'] = np.arange(ret.shape[0])

        return ret

    def _get_node_data(self) -> pd.DataFrame:
        node_data = pd.read_csv(self._node_data_path, sep='\t')

        if os.path.exists(self._derived_path):
            derived = pd.read_csv(self._derived_path, sep='\t')
            node_data = node_data.merge(derived, on='node_id', how='left')

        return node_data

    def _dataloader(self, dataset: BaseDataset, **kwargs: Any
                   ) -> td.DataLoader:
        return td.DataLoader(
            dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            collate_fn = dataset.__collate__,  # *not* self.dataset
            **kwargs
        )

    def train_dataloader(self, **kwargs: Any) -> td.DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs: Any) -> td.DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> td.DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False, **kwargs)

    def compute_split_nodes(self) -> "BaseDataModule":
        self._split_nodes = _train_val_test_split(
            self.dataset.unique_node_ids,
            self.split_proportions,
            self.seed,
        )

        return self

    def split(self) -> "BaseDataModule":
        if self._split_nodes is None:
            self.compute_split_nodes()

        self.post_compute_split_nodes_hook()

        splits = self.dataset.split(splits=self._split_nodes)
        self.train_dataset = splits['train']
        self.val_dataset = splits['val']
        self.test_dataset = splits['test']

        self.post_split_hook()

        return self


#
# Text-only data module
#


class TextDataModule(BaseDataModule):
    def __init__(self,
        tokenizer_name: str,
        mlm: bool = False,
        mlm_probability: float = 0.15,

        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.tokenizer_name = tokenizer_name
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def prepare_data(self) -> "TextDataModule":
        return self  # nothing to do

    def setup(self, stage: Optional[str] = None) -> "TextDataModule":
        self.dataset = TextDataset(
            text = self._get_text(),
            tokenizer_name = self.tokenizer_name,
            mlm = self.mlm,
            mlm_probability = self.mlm_probability,
            device = self.device,
        )

        self.split()

        return self


#
# Graph-only data module
#


class GraphDataModule(BaseDataModule):
    def __init__(self,
        drop_isolates: bool = True,
        directed: bool = False,

        **kwargs: Any
    ) -> None:
        if 'batch_size' in kwargs.keys() and kwargs['batch_size'] != 1:
            raise ValueError('batch_size must be 1 for GraphDataModule')
        elif 'batch_size' not in kwargs.keys():
            kwargs['batch_size'] = 1

        super().__init__(**kwargs)

        self.drop_isolates = drop_isolates
        self.directed = directed

        self._graph_data_cache_path = os.path.join(self.data_dir, 'graph_data.pkl')

    @abstractmethod
    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        raise NotImplementedError()

    def _ensure_graph_object(self) -> None:
        if os.path.exists(self._graph_data_cache_path):
            return

        graph_data = self.prep_graph(
            edgelist = self._get_edgelist(),
            node_data = self._get_node_data()
        )

        with open(self._graph_data_cache_path, 'wb') as f:
            pickle.dump(graph_data, f)

    def _get_graph_object(self) -> Data:
        with open(self._graph_data_cache_path, 'rb') as f:
            ret = pickle.load(f)

        if not self.directed:
            if hasattr(ret, 'num_nodes'):
                num_nodes = ret.num_nodes
            elif hasattr(ret, 'x'):
                num_nodes = ret.x.shape[0]
            else:
                num_nodes = None

            ret.edge_index = to_undirected(ret.edge_index, num_nodes=num_nodes)

        return ret

    def prepare_data(self) -> "GraphDataModule":
        self._ensure_graph_object()

        return self

    def setup(self, stage: Optional[str] = None) -> "BaseDataModule":
        self.dataset = GraphDataset(
            graph_data=self._get_graph_object(),
            drop_isolates=self.drop_isolates,
            device = self.device,
        )

        self.split()

        return self


#
# Graph+text data module
#


class GraphTextDataModule(TextDataModule, GraphDataModule):
    def __init__(self,
        random_data_debug: bool = False,
        max_texts_per_node: int = 0,
        transductive: bool = False,
        transductive_identity_features: bool = False,

        **kwargs: Any
    ) -> None:
        assert not (not transductive and transductive_identity_features), \
            "transductive_identity_features requires transductive = True"

        batch_size = kwargs.pop('batch_size', 32)

        super().__init__(**kwargs)

        self.max_texts_per_node = max_texts_per_node
        self.transductive = transductive
        self.transductive_identity_features = transductive_identity_features

        self.random_data_debug = random_data_debug
        self._real_batch_size = batch_size

    def prepare_data(self) -> "GraphTextDataModule":
        self._ensure_graph_object()

        return self

    def setup(self, stage: Optional[str] = None) -> "GraphTextDataModule":
        self.dataset = GraphTextDataset(
            graph_data = self._get_graph_object(),
            drop_isolates = self.drop_isolates,
            text = self._get_text(),
            tokenizer_name = self.tokenizer_name,
            mlm = self.mlm,
            mlm_probability = self.mlm_probability,
            random_data_debug = self.random_data_debug,
            max_texts_per_node = self.max_texts_per_node,
            transductive = self.transductive,
            transductive_identity_features = self.transductive_identity_features,
            device = self.device,
        )

        self.split()

        # wrap the datasets in batching logic
        datasets = ['train_dataset', 'val_dataset', 'test_dataset']
        for dataset_name in datasets:
            dataset = getattr(self, dataset_name)
            dataset.compute_mutuals()

            dataset = BatchGraphTextDataset(
                dataset,
                batch_size = self._real_batch_size,
                seed = self.seed
            )

            setattr(self, dataset_name, dataset)

        return self


#
# Data-specific prep code
#


class SVDMixin:
    @property
    def svd_vectors_name(self):
        raise NotImplementedError()

    def post_compute_split_nodes_hook(self) -> "SVDMixin":
        svd_dim = 768
        adj = to_dense_adj(self.dataset.graph_data.edge_index).squeeze(dim=0)
        all_nodes = self.dataset.graph_data.node_ids

        train_nodes = self._split_nodes['train']
        train_mask = torch.isin(all_nodes, train_nodes)
        train_adj = adj[train_mask, :][:, train_mask].numpy()

        # use of train_mask on the columns is not an error! need same
        # number of features as for train because we want to project the
        # validation and test data points into the training data's space
        val_nodes = self._split_nodes['val']
        val_mask = torch.isin(all_nodes, val_nodes)
        val_adj = adj[val_mask, :][:, train_mask].numpy()

        test_nodes = self._split_nodes['test']
        test_mask = torch.isin(all_nodes, test_nodes)
        test_adj = adj[test_mask, :][:, train_mask].numpy()

        # FIXME should we use algorithm='arpack' to make this completely
        # deterministic?
        mod = TruncatedSVD(svd_dim, algorithm='randomized').fit(train_adj)
        train_svd = torch.from_numpy(mod.transform(train_adj))
        val_svd = torch.from_numpy(mod.transform(val_adj))
        test_svd = torch.from_numpy(mod.transform(test_adj))

        svd = torch.cat([train_svd, val_svd, test_svd], dim=0)
        svd_nodes = torch.cat([
            all_nodes[train_mask],
            all_nodes[val_mask],
            all_nodes[test_mask]
        ], dim=0)

        # has to be same order as in the dataset object
        svd_inds = (all_nodes[:, None] == svd_nodes).nonzero()[:, 1]

        setattr(
            self.dataset.graph_data,
            self.svd_vectors_name,
            svd[svd_inds, :]
        )

        return self


class TwitterDataMixin(SVDMixin):
    svd_vectors_name = 'tsvd'

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        #
        # Set up the nodes and edges
        #

        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelists don't include isolates
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        #
        # Attributes from our user data
        #

        tmp = node_data.set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        #
        # LM embeddings of bio text on Twitter data
        #

        bios = [
            bio if isinstance(bio, str) else 'No bio provided'
            for bio in node_data['bio'].tolist()
        ]

        lm = st.SentenceTransformer('all-mpnet-base-v2').eval()

        with torch.no_grad():
            embeds = lm.encode(
                bios,
                convert_to_tensor=True,
                normalize_embeddings=False,
                output_value='sentence_embedding',
                device=self.device,
            ).detach().cpu()

        ret = from_networkx(G)

        # order of rows in the tensor attributes is not the same as in G.nodes
        orig = torch.from_numpy(node_data['node_id'].to_numpy())
        inds = (ret.node_ids[:, None] == orig).nonzero()[:, 1]

        ret.x = embeds[inds, ...]

        # this is normally a property, set as an attribute in from_networkx
        # because it can't be inferred from the nonexistent x attribute. now
        # that we've set x, we don't want this and it'll break downstream code
        del ret.num_nodes

        return ret


class PubmedDataMixin(SVDMixin):
    svd_vectors_name = 'x'

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelists don't include isolates
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        tmp = node_data[['node_id', 'label']].set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        ret = from_networkx(G)

        del ret.num_nodes

        return ret


class CoraDataMixin(SVDMixin):
    svd_vectors_name = 'x'

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelists don't include isolates
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        tmp = node_data[['node_id', 'orig_classif', 'cora_ml_classif']]
        tmp = tmp.set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()

        oc = pd.Categorical(tmp['orig_classif'])
        orig_classif_codes = oc.categories.tolist()
        tmp['orig_classif'] = oc.codes

        cmc = pd.Categorical(tmp['cora_ml_classif'])
        cora_ml_classif_codes = cmc.categories.tolist()
        tmp['cora_ml_classif'] = cmc.codes

        nx.set_node_attributes(G, tmp.to_dict('index'))

        ret = from_networkx(G)
        ret.orig_classif_codes = orig_classif_codes
        ret.cora_ml_classif_codes = cora_ml_classif_codes

        del ret.num_nodes

        return ret


class TRexDataMixin(SVDMixin):
    svd_vectors_name = 'x'

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelists don't include isolates
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        node_data['num_categories'] = node_data.sum(axis=1)
        tmp = node_data[['node_id', 'num_categories']].copy().set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        ret = from_networkx(G)

        # order of rows in the tensor attributes is not the same as in G.nodes
        orig = torch.from_numpy(node_data['node_id'].to_numpy())
        inds = (ret.node_ids[:, None] == orig).nonzero()[:, 1]

        node_data = node_data.drop(['node_id', 'num_categories'], axis=1)

        category_names = list(node_data.columns)
        assert all(c.startswith('cat_') for c in category_names)
        category_names = ['Q' + c.replace('cat_', '') for c in category_names]

        node_data = torch.from_numpy(node_data.to_numpy())
        node_data = node_data[inds, :]

        ret.categories = node_data
        ret.category_names = category_names

        del ret.num_nodes

        return ret


#
# Put the data mixins and the other classes together
#


class TwitterGraphDataModule(TwitterDataMixin, GraphDataModule):
    pass


class TwitterGraphTextDataModule(TwitterDataMixin, GraphTextDataModule):
    pass


class PubmedGraphDataModule(PubmedDataMixin, GraphDataModule):
    pass


class PubmedGraphTextDataModule(PubmedDataMixin, GraphTextDataModule):
    pass


class CoraGraphDataModule(CoraDataMixin, GraphDataModule):
    pass


class CoraGraphTextDataModule(CoraDataMixin, GraphTextDataModule):
    pass


class TRexGraphDataModule(TRexDataMixin, GraphDataModule):
    pass


class TRexGraphTextDataModule(TRexDataMixin, GraphTextDataModule):
    pass
