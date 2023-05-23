#!/usr/bin/env python3

import logging

import clip_graph as cg

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.WARNING
    )

    cg.eval.cli()
