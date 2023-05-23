from typing import List, Dict, Any, Optional, Union

import os
import json
import logging
import argparse
import itertools as it

import yaml
import numpy as np
import pandas as pd
import networkx as nx

import sklearn.cluster as cl
import sklearn.metrics as mt

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from networkx.algorithms import community
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from . import utils as ut

logger = logging.getLogger(__name__)

class EvalData:
    def __init__(self,
        comp_nodes_file: str,
        trained_nodes_file: str,
        comp_texts_file: str,
        trained_texts_file: str,
        dataset_yaml_path: str,
        dataset_split: str = 'val',
    ) -> None:
        assert dataset_split in ('train', 'val', 'test')

        super().__init__()

        self.comp_nodes_file = comp_nodes_file
        self.trained_nodes_file = trained_nodes_file
        self.comp_texts_file = comp_texts_file
        self.trained_texts_file = trained_texts_file
        self.dataset_yaml_path = dataset_yaml_path
        self.dataset_split = dataset_split

        self.comp_nodes = torch.load(comp_nodes_file, map_location='cpu')
        self.trained_nodes = torch.load(trained_nodes_file, map_location='cpu')
        self.comp_texts = torch.load(comp_texts_file, map_location='cpu')
        self.trained_texts = torch.load(trained_texts_file, map_location='cpu')

        assert self.trained_nodes.shape[0] == self.comp_nodes.shape[0]
        assert self.trained_texts.shape[0] == self.comp_texts.shape[0]
        assert self.trained_nodes.shape[1] == self.comp_nodes.shape[1]
        assert self.trained_texts.shape[1] == self.comp_texts.shape[1]
        assert self.trained_texts.shape[1] == self.comp_nodes.shape[1]

        dat = ut.datamodule_from_yaml(self.dataset_yaml_path)
        self.seed = dat['seed_everything']
        self.datamodule = dat['dm']
        self.dataset = getattr(dat['dm'], self.dataset_split + '_dataset').dataset

        self.node_mask = (
            self.dataset.graph_data.node_ids == \
            self.dataset.text_node_ids[:, None]
        )

        self.G = to_networkx(self.dataset.graph_data)
        self.G_sims = torch.empty(self.N_nodes, self.N_nodes)

        sims = nx.simrank_similarity(self.G)
        for i in range(self.N_nodes):
            for j in range(self.N_nodes):
                self.G_sims[i][j] = sims[i][j]  # sims is a dict of dicts

    def to(self, device: Union[str, torch.device]):
        self.comp_nodes = self.comp_nodes.to(device)
        self.trained_nodes = self.trained_nodes.to(device)
        self.comp_texts = self.comp_texts.to(device)
        self.trained_texts = self.trained_texts.to(device)
        self.node_mask = self.node_mask.to(device)
        self.G_sims = self.G_sims.to(device)

        return self

    @property
    def inputs(self):
        return {
            'comp_nodes_file': self.comp_nodes_file,
            'trained_nodes_file': self.trained_nodes_file,
            'comp_texts_file': self.comp_texts_file,
            'trained_texts_file': self.trained_texts_file,
            'dataset_yaml_path': self.dataset_yaml_path,
            'dataset_split': self.dataset_split,
        }

    @property
    def N_nodes(self):
        return self.trained_nodes.shape[0]

    @property
    def N_texts(self):
        return self.trained_texts.shape[0]

    @property
    def D_hidden(self):
        return self.trained_texts.shape[1]


class EvalCase:
    defaults = {
        # these work fine, we can compute the exact point estimate in a
        # reasonable period of time
        'eval_topk_accuracy': {
            'n_sims': 100,
            'sample_size': None,
            'resampling_method': 'texts',
        },
        'eval_within_node_dist_pre_post': {
            'n_sims': 100,
            'sample_size': None,
            'resampling_method': 'texts',
        },

        # we can't compute the exact point estimate of these in a reasonable
        # period of time, because they involve a gigantic matrix multiply for
        # cosine similarities on our decent-sized dataset and that's O(n^3).
        # by inspection, a sample size of 5000 produces pretty stable estimates
        # and is quick to run.
        'eval_emb_dist_coupling': {
            'n_sims': 100,
            'sample_size': 5000,
            'resampling_method': 'texts',
        },

        'eval_emb_dist_vs_graph_dist': {
            'n_sims': 100,
            'sample_size': 5000,
            'resampling_method': 'texts',
        },
    }

    #
    # Framework
    #

    def __init__(self,
        data: EvalData,
        what: Optional[List[str]] = None,
        resample: bool = False,
        progress: bool = False,
    ) -> None:
        super().__init__()

        self.data = data
        self.resample = resample
        self.progress = progress

        if what is not None:
            self.what = [
                w if w.startswith('eval_') else 'eval_' + w
                for w in what
            ]
        else:
            self.what = [w for w in dir(self) if w.startswith('eval_')]

    def run(self):
        prog = tqdm if self.progress else lambda x: x
        prog = prog(self.what)

        stats = {}
        for w in prog:
            if self.progress:
                prog.set_description(w)

            defaults = self.defaults[w]

            ret = {}
            if defaults['sample_size'] is None:
                func = getattr(self, w)
                ret['exact_point'] = func()
            else:
                # some inter-text similarity operations don't fit into
                # memory, so we have to sample them to calculate them
                # at all (without, sure, more work than it's worth rn to
                # batch up and calculate the whole thing)
                ret['approx_point'] = self.run_resampling(
                    funcname = w,
                    how = defaults['resampling_method'],
                    n_sims = 1,
                    sample_size = defaults['sample_size'],
                )[0]

            if self.resample:
                ret['resample'] = self.run_resampling(
                    funcname = w,
                    how = defaults['resampling_method'],
                    n_sims = defaults['n_sims'],
                    sample_size = defaults['sample_size'],
                )

            stats[w] = ret

        return dict(stats, **self.data.inputs)

    # the sample_size is because there may be many texts and operations we
    # might want to do on the whole text-text similarity matrix may not fit
    # into memory, so we need to downsample the resample
    def run_resampling(self,
        funcname: str,
        how: str = 'texts',
        n_sims: int = 100,
        sample_size: Optional[int] = None
    ) -> List[Dict[str, Union[int, float]]]:
        assert how in ('texts', 'nodes')

        prog = tqdm if self.progress else lambda x: x

        if sample_size is None:
            if how == 'texts':
                sample_size = self.data.N_texts
            else:  # how == 'nodes'
                sample_size = self.data.N_nodes

        runs = []
        for i in prog(range(n_sims)):
            if how == 'texts':
                text_inds = torch.multinomial(
                    torch.ones((self.data.N_texts,)),
                    sample_size,
                    replacement=True
                )

                node_inds = self.data.node_mask[text_inds, :].nonzero()[:, 1]
            else:  # how == 'nodes'
                text_inds = None

                node_inds = torch.multinomial(
                    torch.ones((self.data.N_nodes,)),
                    sample_size,
                    replacement=True
                )

            func = getattr(self, funcname)
            tmp = func(text_inds, node_inds)
            if isinstance(tmp, dict):
                tmp = [tmp]

            tmp = [dict(d, run=i) for d in tmp]
            runs += tmp

        return runs

    #
    # Top-k accuracy
    #

    def eval_topk_accuracy(self,
        text_inds: Optional[torch.Tensor] = None,
        node_inds: Optional[torch.Tensor] = None,
        max_k: int = 10,
    ) -> List[Dict[str, Union[int, float]]]:
        assert ut.count(text_inds, node_inds) in (0, 2)

        trained_texts = self.data.trained_texts
        if text_inds is not None:
            trained_texts = trained_texts[text_inds, ...]

        comp_texts = self.data.comp_texts
        if text_inds is not None:
            comp_texts = comp_texts[text_inds, ...]

        node_mask = self.data.node_mask
        if text_inds is not None:
            node_mask = node_mask[text_inds, ...]

        trained_sims = ut.cos_sim(trained_texts, self.data.trained_nodes)
        comp_sims = ut.cos_sim(comp_texts, self.data.comp_nodes)

        ret = []

        for k in range(1, max_k + 1):
            trained_acc = ut.topk_accuracy(node_mask, trained_sims, k=k)
            comp_acc = ut.topk_accuracy(node_mask, comp_sims, k=k)

            ret += [{
                'k': int(k),
                'trained_acc': float(trained_acc),
                'comp_acc': float(comp_acc),
                'diff_acc': float(trained_acc - comp_acc),
            }]

        return ret

    #
    # Within-node pre=>post change in average text similarity
    #

    def eval_within_node_dist_pre_post(self,
        text_inds: Optional[torch.Tensor] = None,
        node_inds: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        assert ut.count(text_inds, node_inds) in (0, 2)

        trained_texts = self.data.trained_texts
        if text_inds is not None:
            trained_texts = trained_texts[text_inds, ...]

        comp_texts = self.data.comp_texts
        if text_inds is not None:
            comp_texts = comp_texts[text_inds, ...]

        node_mask = self.data.node_mask
        if text_inds is not None:
            node_mask = node_mask[text_inds, ...]

        # speeds up the loop below
        trained_texts = torch.clone(trained_texts)
        comp_texts = torch.clone(comp_texts)
        node_mask = torch.clone(node_mask)

        runs = []
        for j in range(self.data.N_nodes):
            inds = node_mask[:, j].nonzero().squeeze()
            if inds.numel() in (0, 1):
                continue

            trained_sim = ut.cos_sim(trained_texts[inds, :]).mean().item()
            comp_sim = ut.cos_sim(comp_texts[inds, :]).mean().item()

            runs += [{
                'node_ind': j,
                'trained_sim': trained_sim,
                'comp_sim': comp_sim,
            }]

        if len(runs) > 0:
            trained_sim_avg = sum(r['trained_sim'] for r in runs) / len(runs)
            comp_sim_avg = sum(r['comp_sim'] for r in runs) / len(runs)
        else:
            trained_sim_avg, comp_sim_avg = np.nan, np.nan

        return {
            'trained_sim_avg': float(trained_sim_avg),
            'comp_sim_avg': float(comp_sim_avg),
            'diff_sim_avg': float(trained_sim_avg - comp_sim_avg),
        }

    #
    # coupling of text/graph embedding distances
    #

    def eval_emb_dist_coupling(self,
        text_inds: Optional[torch.Tensor] = None,
        node_inds: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        assert ut.count(text_inds, node_inds) in (0, 2)

        trained_texts = self.data.trained_texts
        if text_inds is not None:
            trained_texts = trained_texts[text_inds, ...]

        comp_texts = self.data.comp_texts
        if text_inds is not None:
            comp_texts = comp_texts[text_inds, ...]

        trained_nodes = self.data.trained_nodes
        if node_inds is not None:
            trained_nodes = trained_nodes[node_inds, ...]

        comp_nodes = self.data.comp_nodes
        if node_inds is not None:
            comp_nodes = comp_nodes[node_inds, ...]

        trained_text_sims = ut.cos_sim(trained_texts)
        trained_node_sims = ut.cos_sim(trained_nodes)
        comp_text_sims = ut.cos_sim(comp_texts)
        comp_node_sims = ut.cos_sim(comp_nodes)

        trained_corr = torch.corrcoef(torch.stack([
            trained_text_sims.flatten(),
            trained_node_sims.flatten(),
        ], dim=0))[0, 1].item()

        comp_corr = torch.corrcoef(torch.stack([
            comp_text_sims.flatten(),
            comp_node_sims.flatten(),
        ], dim=0))[0, 1].item()

        return {
            'trained_corr': float(trained_corr),
            'comp_corr': float(comp_corr),
            'diff_corr': float(trained_corr - comp_corr),
        }

    #
    # Embedding distance vs unembedded graph distance
    #

    def eval_emb_dist_vs_graph_dist(self,
        text_inds: Optional[torch.Tensor] = None,
        node_inds: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        assert ut.count(text_inds, node_inds) in (0, 2)

        trained_texts = self.data.trained_texts
        if text_inds is not None:
            trained_texts = trained_texts[text_inds, ...]

        comp_texts = self.data.comp_texts
        if text_inds is not None:
            comp_texts = comp_texts[text_inds, ...]

        trained_nodes = self.data.trained_nodes
        if node_inds is not None:
            trained_nodes = trained_nodes[node_inds, ...]

        comp_nodes = self.data.comp_nodes
        if node_inds is not None:
            comp_nodes = comp_nodes[node_inds, ...]

        G_sims = self.data.G_sims
        if node_inds is not None:
            G_sims = G_sims[node_inds, ...][..., node_inds]

        trained_text_sims = ut.cos_sim(trained_texts)
        comp_text_sims = ut.cos_sim(comp_texts)
        trained_node_sims = ut.cos_sim(trained_nodes)
        comp_node_sims = ut.cos_sim(comp_nodes)

        trained_text_corr = torch.corrcoef(torch.stack([
            trained_text_sims.flatten(),
            G_sims.flatten(),
        ], dim=0))[0, 1].item()

        comp_text_corr = torch.corrcoef(torch.stack([
            comp_text_sims.flatten(),
            G_sims.flatten(),
        ], dim=0))[0, 1].item()

        trained_node_corr = torch.corrcoef(torch.stack([
            trained_node_sims.flatten(),
            G_sims.flatten(),
        ], dim=0))[0, 1].item()

        comp_node_corr = torch.corrcoef(torch.stack([
            comp_node_sims.flatten(),
            G_sims.flatten(),
        ], dim=0))[0, 1].item()

        return {
            'trained_text_corr': float(trained_text_corr),
            'comp_text_corr': float(comp_text_corr),
            'diff_text_corr': float(trained_text_corr - comp_text_corr),
            'trained_node_corr': float(trained_node_corr),
            'comp_node_corr': float(comp_node_corr),
            'diff_node_corr': float(trained_node_corr - comp_node_corr),
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='evaluate model performance')
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    cased = subparsers.add_parser('case')
    cased.add_argument('--trained-texts', required=True,
                       help='model text preds')
    cased.add_argument('--comp-texts', required=True,
                       help='comparison text preds')
    cased.add_argument('--trained-nodes', required=True,
                       help='model node preds')
    cased.add_argument('--comp-nodes', required=True,
                       help='comparison node preds')
    cased.add_argument('-c', '--dataset-yaml-path', required=True,
                       help='yaml configuration file for eval dataset')
    cased.add_argument('-o', '--output-path',
                       help='output path (default: stdout)')

    batch = subparsers.add_parser('batch')
    batch.add_argument('-f', '--file', required=True,
                       help='file containing list of cases to evaluate')
    batch.add_argument('--in-dir', default='data/embeds/',
                       help='input directory prefix')
    batch.add_argument('--out-dir', default='data/evals/',
                       help='output directory prefix')

    for sp in [cased, batch]:
        sp.add_argument('-s', '--dataset-split', default='test',
                        help='datamodule split used in scores (default "val")')
        sp.add_argument('-w', '--what', nargs='*',
                        help='what evaluations to run (default: all)')
        sp.add_argument('-r', '--resample', action='store_true',
                        help='boostrap resampling of metrics')

        sp.add_argument('-d', '--device', default='cpu',
                        help='pytorch device string for models')
        sp.add_argument('-p', '--progress', action='store_true',
                        help='show progress bars')

    return parser.parse_args()


def process_batch_file(cases, in_dir='data/embeds/', out_dir='data/evals/'):
    ret = []
    for dataset_name, dataset in cases.items():
        if dataset_name == 'baselines':
            continue

        for _, task in dataset.items():
            for comp in task['comparisons']:
                obj = {'dataset_yaml_path': task['dataset_yaml_path']}

                for arg in comp:
                    if arg == 'slug':
                        obj['output_path'] = os.path.join(out_dir, comp[arg]) + '.json'
                    else:
                        obj[arg + '_file'] = os.path.join(in_dir, comp[arg])

                ret += [obj]

    for obj in ret:
        cols = ['dataset_yaml_path', 'output_path', 'comp_texts_file',
                'comp_nodes_file', 'trained_texts_file', 'trained_nodes_file']
        assert set(cols) == set(obj.keys())

    return ret


def cli() -> None:
    args = parse_args()

    if args.subcommand == 'batch':
        with open(args.file, 'rt') as f:
            cases = process_batch_file(
                yaml.safe_load(f),
                args.in_dir,
                args.out_dir
            )

        os.makedirs(args.out_dir, exist_ok=True)
    else:
        cases = [{
            'dataset_yaml_path': args.dataset_yaml_path,
            'output_path': args.output_path,
            'comp_texts_file': args.comp_texts,
            'comp_nodes_file': args.comp_nodes,
            'trained_texts_file': args.tarined_texts,
            'trained_nodes_file': args.trained_nodes,
        }]

        os.makedirs(
            os.path.abspath(os.path.dirname(args.output_path)),
            exist_ok=True,
        )

    with tqdm(cases, disable=(not args.progress)) as pbar:
        for cse in pbar:
            output_path = cse.pop('output_path')
            cse['dataset_split'] = args.dataset_split

            data = EvalData(**cse).to(args.device)
            pl.seed_everything(data.seed, workers=True)

            stats = EvalCase(
                data=data,
                what=args.what,
                resample=args.resample,
                progress=args.progress,
            ).run()

            with open(output_path, 'wt') as f:
                json.dump(stats, f)
