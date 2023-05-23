'''
Utilities for clip-graph
'''

from typing import Dict, Any, Optional, List, Union, Tuple

import sys
import bz2
import gzip
import importlib
import contextlib

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from omegaconf import OmegaConf


def coalesce(*args: Any) -> Optional[Any]:
    "Return the first argument which is not None (like SQL's COALESCE)."

    try:
        return next(filter(lambda x: x is not None, args))
    except StopIteration:
        return None


def count(*args: Any) -> int:
    "Count the number of arguments which are not None."

    return sum([s is not None for s in args])


def import_class(path: str) -> Any:
    "Import and return a class given in dotted path notation."

    *mod, kls = path.split('.')

    mod = importlib.import_module('.'.join(mod))
    kls = getattr(mod, kls)

    return kls


def datamodule_from_yaml(file: str) -> pl.LightningDataModule:
    conf = OmegaConf.load(file)

    kls = import_class(conf.data.class_path)
    kls_args = OmegaConf.to_object(conf.data.init_args)

    dm = kls(**kls_args)

    dm = dm.prepare_data()
    dm = dm.setup()

    return {
        'dm': dm,
        'seed_everything': conf.seed_everything,
    }


def model_from_yaml(file: str) -> torch.nn.Module:
    conf = OmegaConf.load(file)

    kls = import_class(conf.model.class_path)
    kls_args = OmegaConf.to_object(conf.model.init_args)

    return kls(**kls_args)


def cos_sim(
    c1: torch.Tensor,
    c2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    c1 = F.normalize(c1, p=2, dim=1)

    if c2 is None:
        c2 = c1
    else:
        c2 = F.normalize(c2, p=2, dim=1)

    return c1 @ c2.T


def topk_accuracy(
    node_mask: torch.Tensor,
    sims: torch.Tensor,
    k: int = 1
) -> float:
    true = node_mask.long().argmax(dim=1)
    pred = sims.topk(k, dim=1)[1]

    return (true[:, None] == pred).any(-1).nonzero().shape[0] / true.shape[0]


def argunique(
    x: torch.Tensor,
    sort: bool = True,
    dim: int = None
) -> torch.Tensor:
    u, inv = torch.unique(x, return_inverse=True, dim=dim)

    edim = dim if dim is not None else 0

    perm = torch.arange(inv.shape[edim], dtype=inv.dtype, device=inv.device)
    inv, perm = inv.flip([edim]), perm.flip([edim])
    inds = inv.new_empty(u.shape[edim]).scatter_(edim, inv, perm)

    if sort:
        inds = inds[torch.argsort(inds)]

    return inds


def _train_val_test_split(
    node_ids: torch.Tensor,
    split_proportions: List[Union[float, int]],
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    # shuffle the node_ids
    node_ids = node_ids[torch.randperm(node_ids.shape[0], generator=gen)]

    # NOTE sum(split_counts) may be <= 2 elements shorter than
    # len(node_ids), but this is fine; we just apportion all remaining
    # elements to the test set. so: train and val set lengths are
    # truncated from float to int, test set length is all remaining
    # elements.
    split_counts = (split_proportions * node_ids.shape[0]).long()
    n_train, n_val, n_test = split_counts

    train_nodes = node_ids[:n_train]
    val_nodes = node_ids[n_train:(n_train + n_val)]
    test_nodes = node_ids[(n_train + n_val):]

    return {
        'train': train_nodes,
        'val': val_nodes,
        'test': test_nodes,
    }


#
# One function you can pass a filename to and get a file-like object, including
# handling of '-' for stdin/stdout or gz/bz2
#


def zip_safe_open(f, *args, **kwargs):
    if f.lower().endswith('.gz'):
        func = gzip.open
    elif f.lower().endswith('.bz2'):
        func = bz2.open
    else:
        func = open

    return func(f, *args, **kwargs)


@contextlib.contextmanager
def smart_open(filename='-', mode='r', method=zip_safe_open, **kwargs):
    is_read = (mode[0] == 'r')

    if filename and filename != '-':
        fh = method(filename, mode, **kwargs)
    elif is_read:
        fh = sys.stdin
    else:
        # it doesn't make sense to ask for read on stdout or write on
        # stdin, so '-' can be unambiguously resolved to one or the other
        # depending on the open mode
        fh = sys.stdout

    try:
        yield fh
    finally:
        if is_read and fh is not sys.stdin:
            fh.close()
        elif not is_read and fh is not sys.stdout:
            fh.close()
