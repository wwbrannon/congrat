ConGraT
---

This repo contains code for the paper [ConGraT: Self-Supervised Contrastive
Pretraining for Joint Graph and Text
Embeddings](https://arxiv.org/abs/2305.14321).

For our updated Pubmed dataset and scripts, see [this
repo](https://github.com/mit-ccc/pubmed-dataset).

**What we're trying to do**: We want to learn a single model of the joint
distribution of text and a graph structure, where the graph is over the entities
generating the text. (This latter condition is what distinguishes our case from
models using knowledge graphs.) This problem occurs in a variety of settings:
the follow graph among users who post tweets, link graphs between web pages,
citation networks for academic articles, etc. We view it as a kind of multimodal
learning, allowing models to leverage graph data that co-occurs with texts.
