#!/usr/bin/env python3

import os
import logging

import torch
import pytorch_lightning.cli as cl

import clip_graph as cg

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.WARNING
    )

    # avoid a whole bunch of annoying error messages
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    # use tensor cores appropriately
    torch.set_float32_matmul_precision('high')

    cl.LightningCLI(
        model_class=cg.LitBase,
        subclass_mode_model=True,

        datamodule_class=cg.BaseDataModule,
        subclass_mode_data=True,

        parser_kwargs={
            'error_handler': None,  # print noisy tracebacks
            'parser_mode': 'omegaconf',  # better variable interpolation
        }
    )
