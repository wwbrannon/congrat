#!/usr/bin/env python3

import os
import logging

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

    cg.scoring.cli()
