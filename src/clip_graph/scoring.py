from typing import List, Dict, Any, Optional

import os
import re
import logging
import argparse
import subprocess as sp

import yaml

import torch

import pytorch_lightning as pl
from .data.dataset import BatchGraphTextDataset

from tqdm import tqdm

from . import utils as ut

logger = logging.getLogger(__name__)


def interpret_ckpt_dir(
    ckpt_dir: str,
    dm: Optional[pl.LightningDataModule] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    ## Load the checkpoint configuration
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'rt') as f:
        config = yaml.safe_load(f)

    ## Set the seed from config
    seed = int(config['seed_everything'])
    pl.seed_everything(seed)

    ## Load the checkpoint datamodule if we weren't given one
    if dm is None:
        kls = config['data']['class_path']
        kls = ut.import_class(kls)

        init_args = config['data']['init_args']

        dm = kls(**init_args)
        dm.setup()

    ## What's the model class to use?
    kls = ut.import_class(config['model']['class_path'])

    ## Identify the checkpoint to use
    def checkpoint_key(s):
        match = re.match('epoch=([0-9]+)-step=([0-9]+).ckpt', s)
        match = match.groups()
        match = tuple([int(c) for c in match])

        return match

    checkpoints = os.listdir(os.path.join(ckpt_dir, 'checkpoints'))
    checkpoints = sorted(checkpoints, key=checkpoint_key, reverse=True)

    checkpoint = checkpoints[0]  # most recent checkpoint if > 1
    checkpoint = os.path.join(ckpt_dir, 'checkpoints', checkpoint)

    ## Load the checkpoint - note we have to handle deepspeed separately
    if not os.path.isdir(checkpoint):  # is normal PL checkpoint
        model = kls.load_from_checkpoint(checkpoint)
    else:  # is deepspeed checkpoint
        sd_path = os.path.join(checkpoint, 'state_dict.pt')

        if not os.path.exists(sd_path):
            conv_script = os.path.join(checkpoint, 'zero_to_fp32.py')
            sp.check_call([conv_script, checkpoint, sd_path], cwd=checkpoint)

        with open(sd_path, 'rb') as f:
            sd = torch.load(f)

        # deepspeed saves the keys with a 'module.' in front
        sd = {'.'.join(k.split('.')[1:]) : v for k, v in sd.items()}

        model = kls(**config['model']['init_args'])

        # the LM heads for models don't get saved for some reason, but we don't
        # care about that because we don't use them for this scoring
        status = model.load_state_dict(sd, strict=False)
        mk, uk = status.missing_keys, status.unexpected_keys

        if len(mk) > 0:
            logger.warning(f'Missing keys from state_dict: {",".join(mk)}')

        if len(uk) > 0:
            logger.warning(f'Unexpected keys from state_dict: {",".join(uk)}')

    model = model.eval().to(device)

    return {
        'dm': dm,
        'model': model,
        'config': config,
        'device': device,
    }


def get_gnn_scores(
    dat: Dict[str, Any],
    split: str = 'val',
    gnn_keys: List[str] = ['graph_x'],
) -> torch.Tensor:
    assert split in ('train', 'val', 'test')

    tmp = getattr(dat['dm'], split + '_dataloader')()
    tmp = next(iter(tmp))

    vei = tmp['graph_edge_index'].to(dat['device'])

    vx = [tmp[k].float() for k in gnn_keys]
    vx = torch.cat(vx, dim=1).to(dat['device'])

    if hasattr(dat['model'], 'embed_nodes') and callable(getattr(dat['model'], 'embed_nodes')):
        func = getattr(dat['model'], 'embed_nodes')
    else:
        func = getattr(dat['model'], '__call__')

    with torch.no_grad():
        ret = func(vx, vei)

    if not isinstance(ret, dict):
        return ret
    elif ret['output'] is not None:
        return ret['output']
    else:
        return ret['last_hidden_state']


def get_lm_scores(
    dat: Dict[str, Any],
    split: str = 'val',
    progress: bool = False
) -> torch.Tensor:
    assert split in ('train', 'val', 'test')

    ds = getattr(dat['dm'], split + '_dataset')
    if isinstance(ds, BatchGraphTextDataset):
        ds = ds.dataset

        # in this case, dm.val_dataset and its loader serve batches,
        # not individual examples
        ldr = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=128,
            collate_fn=ds.__collate__,
        )
    else:
        ldr = getattr(dat['dm'], split + '_dataloader')()

    if hasattr(dat['model'], 'embed_text') and callable(getattr(dat['model'], 'embed_text')):
        func = getattr(dat['model'], 'embed_text')
    else:
        func = getattr(dat['model'], '__call__')

    if progress:
        ldr = tqdm(ldr)

    scores, node_ids = [], []
    for batch in ldr:
        ii = batch['input_ids'].to(dat['device'])
        am = batch['attention_mask'].to(dat['device'])

        with torch.no_grad():
            scores += [func(ii, am)]

        node_ids += [batch['text_node_ids']]

    node_ids = torch.cat(node_ids, dim=0)
    assert (node_ids == ds.text_node_ids).all()
    del node_ids

    return torch.cat(scores, dim=0)


def score_lm_pretrain(ckpt_dir, dm=None, device='cpu', split='val',
                      progress=False, pooling_mode=None, normalize=None):
    assert split in ('train', 'val', 'test')

    dat = interpret_ckpt_dir(ckpt_dir, dm, device)
    dat['model'] = dat['model'].model

    dat['model'] = dat['model'].to_sentence_embedding(
        pooling_mode = pooling_mode,
        normalize = normalize,
    )

    return {
        'text': get_lm_scores(dat, split=split, progress=progress),
    }


def score_gnn_pretrain(
    ckpt_dir: str,
    dm: Optional[pl.LightningDataModule] = None,
    device: str = 'cpu',
    split: str = 'val',
) -> Dict[str, torch.Tensor]:
    assert split in ('train', 'val', 'test')

    dat = interpret_ckpt_dir(ckpt_dir, dm, device)
    dat['model'] = dat['model'].model.encoder

    try:
        gnn_keys = dat['config']['model']['init_args']['node_features_keys']
    except KeyError:
        logger.warning('No node_features_keys, defaulting to graph_x')
        gnn_keys = ['graph_x']

    return {
        'graph': get_gnn_scores(dat, split=split, gnn_keys=gnn_keys),
    }


def score_clip_graph(
    ckpt_dir: str,
    dm: Optional[pl.LightningDataModule] = None,
    device: str = 'cpu',
    split: str = 'val',
    progress: bool = False
) -> Dict[str, torch.Tensor]:
    assert split in ('train', 'val', 'test')

    dat = interpret_ckpt_dir(ckpt_dir, dm, device)
    dat['model'] = dat['model'].model

    try:
        gnn_keys = dat['config']['model']['init_args']['gnn_node_features_keys']
    except KeyError:
        logger.warning('No gnn_node_features_keys, defaulting to graph_x')
        gnn_keys = ['graph_x']

    return {
        'graph': get_gnn_scores(dat, split=split, gnn_keys=gnn_keys),
        'text': get_lm_scores(dat, split=split, progress=progress),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='compute scores on data')
    sp = parser.add_subparsers(dest='subcommand')

    subcommands = {
        'lm_pretrain': sp.add_parser('lm_pretrain', description='score texts'),
        'gnn_pretrain': sp.add_parser('gnn_pretrain', description='score graph'),
        'clip_graph': sp.add_parser('clip_graph', description='score clip-graph'),
    }

    for k, v in subcommands.items():
        v.set_defaults(func=globals()['score_' + k])

        v.add_argument('-i', '--input-dir', required=True,
                       help='directory for model version')
        v.add_argument('-o', '--output-dir', required=True,
                       help='output directory')
        v.add_argument('-s', '--split', default='val',
                       help='datamodule split to score (default "val")')
        v.add_argument('-d', '--device', default='cpu',
                       help='pytorch device string for models')
        v.add_argument('-c', '--dataset-yaml-path',
                       help='yaml configuration file for eval dataset '
                            '(default: score on the requested split of the '
                            'datamodule used for training)')

    subcommands['lm_pretrain'].add_argument(
            '-p', '--progress', action='store_true',
            help='display progress bars'
    )

    subcommands['clip_graph'].add_argument(
            '-p', '--progress', action='store_true',
            help='display progress bars'
    )

    subcommands['lm_pretrain'].add_argument(
        '-m', '--pooling-mode',
        default='mean', choices=['mean', 'cls', 'max'],
        help='pooling mode for pretrained language model'
    )

    subcommands['lm_pretrain'].add_argument(
        '-n', '--no-normalize', action='store_true',
        help="skip normalizing pooled output vectors for language model"
    )

    return parser.parse_args()


def cli() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_yaml_path is not None:
        dm = ut.datamodule_from_yaml(args.dataset_yaml_path)['dm']

    inpath = os.path.abspath(args.input_dir)

    kwargs = {
        'ckpt_dir': inpath,
        'split': args.split,
        'device': args.device,
    }

    if args.dataset_yaml_path is not None:
        kwargs['dm'] = dm

    if args.subcommand in ('lm_pretrain', 'clip_graph'):
        kwargs['progress'] = args.progress

    if args.subcommand == 'lm_pretrain':
        kwargs['pooling_mode'] = args.pooling_mode
        kwargs['normalize'] = not args.no_normalize

    scores = args.func(**kwargs)

    for k, v in scores.items():
        outpath = f'{args.split}-{k}-{inpath[1:].replace("/", "_")}.pt'
        outpath = os.path.join(args.output_dir, outpath)
        outpath = os.path.abspath(outpath)

        torch.save(v.cpu(), outpath)
