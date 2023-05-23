#!/usr/bin/env python3

#
# First fetch the original data files as described in orig/README.md
#

import os
import re
import bz2
import csv
import gzip
import json
import time
import random
import zipfile

import networkx as nx

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from pandasql import sqldf

if __name__ == '__main__':
    #
    # T-REx data
    #

    text_fieldnames = ['node_id', 'kind', 'text']
    graph_fieldnames = ['source', 'target', 'start', 'end']

    os.makedirs('raw', exist_ok=True)
    os.makedirs('intermediate', exist_ok=True)

    with gzip.open('intermediate/text-full.csv.gz', 'wt') as f:
        csv.writer(f).writerow(text_fieldnames)

    with gzip.open('intermediate/graph-full.csv.gz', 'wt') as f:
        csv.writer(f).writerow(graph_fieldnames)

    with open('orig/TREx.zip', 'rb') as zfile:
        zf = zipfile.ZipFile(zfile)

        for file in tqdm(zf.namelist()):
            text, graph = [], []

            with zf.open(file, 'r') as f:
                obj = json.load(f)

            for doc in obj:
                docid = re.search('Q[0-9]+$', doc['docid']).group(0)

                text += [(docid, 'title', doc['title'])]
                text += [(docid, 'abstract', doc['text'])]

                for et in doc['entities']:
                    if et['annotator'] == 'Wikidata_Spotlight_Entity_Linker':
                        graph += [(
                            docid,
                            re.search('Q[0-9]+$', et['uri']).group(0),
                            et['boundaries'][0],
                            et['boundaries'][1],
                        )]

            with gzip.open('intermediate/text-full.csv.gz', 'at') as f:
                csv.writer(f).writerows(text)

            with gzip.open('intermediate/graph-full.csv.gz', 'at') as f:
                csv.writer(f).writerows(graph)

    #
    # Category hierarchy
    #

    BROADER_URL = 'http://www.w3.org/2004/02/skos/core#broader'
    CAT_URL_PREFIX = 'http://dbpedia.org/resource/Category:'

    with gzip.open('intermediate/category_tree.csv.gz', 'wt') as f:
        fieldnames = ['cat', 'parent']

        writer = csv.writer(f)
        writer.writerow(fieldnames)

        with bz2.open('orig/categories_lang=en_skos.ttl.bz2', 'rt') as f:
            for line in tqdm(f):
                if line.startswith('#'):
                    continue

                if not BROADER_URL in line:
                    continue

                cat, rel, parent, _ = line.split('>')
                cat = cat.strip().lstrip('<')
                rel = rel.strip().lstrip('<')
                parent = parent.strip().lstrip('<')

                assert rel == BROADER_URL
                assert cat.startswith(CAT_URL_PREFIX)
                assert parent.startswith(CAT_URL_PREFIX)

                cat = cat.replace(CAT_URL_PREFIX, '')
                parent = parent.replace(CAT_URL_PREFIX, '')

                writer.writerow([cat, parent])

    #
    # Entity to Q ID mapping
    #

    with gzip.open('intermediate/entity_map.csv.gz', 'wt') as f:
        fieldnames = ['qid', 'page', 'domain']

        writer = csv.writer(f)
        writer.writerow(fieldnames)

        with bz2.open('orig/sameas-all-wikis.ttl.bz2', 'rt') as f:
            for line in tqdm(f):
                if line.startswith('#'):
                    continue

                qid, rel, page, _ = line.split()
                assert rel == '<http://www.w3.org/2002/07/owl#sameAs>'

                page = page.lstrip('<').rstrip('>')
                domain = re.sub('http://([a-z.]*?dbpedia.org)/resource/.*',
                                lambda s: s.group(1), page)
                page = re.sub('http://([a-z]*.)?dbpedia.org/resource/', '', page)

                qid = qid.lstrip('<').rstrip('>')
                assert qid.startswith('http://wikidata.dbpedia.org/resource/')
                qid = qid.replace('http://wikidata.dbpedia.org/resource/', '')

                writer.writerow([qid, page, domain])

    #
    # Article <=> category assignments
    #

    SUBJECT_REL_URL = 'http://purl.org/dc/terms/subject'
    PAGE_URL_PREFIX = 'http://dbpedia.org/resource/'

    with gzip.open('intermediate/article_category.csv.gz', 'wt') as f:
        fieldnames = ['page', 'cat']

        writer = csv.writer(f)
        writer.writerow(fieldnames)

        with bz2.open('orig/categories_lang=en_articles.ttl.bz2', 'rt') as f:
            for line in tqdm(f):
                if line.startswith('#'):
                    continue

                page, rel, cat, _ = line.split()
                page = page.lstrip('<').rstrip('>')
                rel = rel.lstrip('<').rstrip('>')
                cat = cat.lstrip('<').rstrip('>')

                assert rel == SUBJECT_REL_URL

                page = page.replace(PAGE_URL_PREFIX, '')
                cat = cat.replace(PAGE_URL_PREFIX + 'Category:', '')

                writer.writerow([page, cat])

    #
    # Prepare link tables
    #

    # Prep: reload entity map
    with gzip.open('intermediate/entity_map.csv.gz', 'rt') as emf:
        entity_map = pd.read_csv(emf)

    entity_map['qid'] = entity_map['qid'].str.replace('Q', '').astype(int)
    entity_map['is_category'] = entity_map['page'].str.startswith('Category:')
    entity_map = entity_map.loc[~entity_map['is_category'].isna(), :]

    # Category <=> Q ID
    cat_map = entity_map.loc[
        entity_map['is_category'] &
        entity_map.domain.isin(['dbpedia.org', 'commons.dbpedia.org']),
    :].copy()
    cat_map['page'] = cat_map['page'].str.replace('Category:', '')
    cat_map = cat_map.drop(['domain', 'is_category'], axis=1).drop_duplicates()

    with gzip.open('intermediate/category_map.csv.gz', 'wt') as cmf:
        cat_map \
            .rename({'qid': 'cat_id', 'page': 'cat'}, axis=1) \
            .to_csv(cmf, index=False)

    # Article <=> Q ID
    art_map = entity_map.loc[
        ~entity_map['is_category'] &
        entity_map.domain.isin(['dbpedia.org', 'commons.dbpedia.org']),
    :].copy()
    art_map['page'] = art_map['page'].str.replace('Category:', '')
    art_map = art_map.drop(['domain', 'is_category'], axis=1).drop_duplicates()

    with gzip.open('intermediate/article_map.csv.gz', 'wt') as amf:
        art_map \
            .rename({'qid': 'page_id'}, axis=1) \
            .to_csv(amf, index=False)

    # Article Q ID <=> Category Q ID
    with gzip.open('intermediate/article_category.csv.gz', 'rt') as f:
        art_cat = pd.read_csv(f)

    assert art_cat.shape[0] == art_cat.drop_duplicates().shape[0]

    art_cat = art_cat \
        .merge(cat_map.rename({'qid': 'cat_id', 'page': 'cat'}, axis=1), left_on='cat', right_on='cat') \
        .merge(art_map.rename({'qid': 'page_id'}, axis=1), left_on='page', right_on='page') \
        [['cat_id', 'page_id']] \
        .drop_duplicates()

    with gzip.open('intermediate/article_category_map.csv.gz', 'wt') as f:
        art_cat.to_csv(f, index=False)

    # Category hierarchy as (cat_id, parent_id)
    with gzip.open('intermediate/category_tree.csv.gz', 'rt') as f:
        cat_tree = pd.read_csv(f)

    cat_tree = cat_tree \
        .merge(cat_map.rename({'qid': 'cat_id'}, axis=1), left_on='cat', right_on='page') \
        .merge(cat_map.rename({'qid': 'parent_id'}, axis=1), left_on='parent', right_on='page') \
        [['cat_id', 'parent_id']] \
        .drop_duplicates()

    cat_tree = pd.concat([
        cat_tree,

        cat_map \
            .loc[~cat_map['qid'].isin(cat_tree['cat_id']), :] \
            .drop('page', axis=1) \
            .rename({'qid': 'cat_id'}, axis=1) \
            .assign(parent_id=-1)
    ], axis=0)

    with gzip.open('intermediate/category_tree_map.csv.gz', 'wt') as f:
        cat_tree.to_csv(f, index=False)

    #
    # Select and write out the robots category
    #

    with gzip.open('intermediate/graph-full.csv.gz', 'rt') as f:
        adj = pd.read_csv(f)

    # can be >1 link to the same target in an abstract
    adj = adj.drop(['start', 'end'], axis=1).drop_duplicates()

    # no self-loops
    adj = adj.loc[adj['source'] != adj['target'], :]

    # save some memory
    adj['source'] = adj['source'].str.replace('Q', '').astype(int)
    adj['target'] = adj['target'].str.replace('Q', '').astype(int)

    def subcats(initial_id, max_depth=10):
        query = '''
        with recursive ct as
        (
            select
                {initial_id} as cat_id,
                -1 as parent_id,
                0 as level

            union all

            select
                ctr.cat_id,
                ctr.parent_id,
                ct.level + 1 as level
            from cat_tree ctr
                inner join ct on ctr.parent_id = ct.cat_id
            where
                ct.level + 1 <= {max_depth}  -- just in case; no infinite loops
        )

        select
            ct.cat_id,
            ct.parent_id,
            ct.level
        from ct;
        '''

        query = query.format(initial_id=initial_id, max_depth=max_depth)

        return sqldf(query, globals())

    target_cat_id = 8670893  # Category:Robots

    cats = subcats(target_cat_id)
    pages = art_cat.loc[art_cat['cat_id'].isin(cats['cat_id']), 'page_id'].unique()
    edges = adj.loc[adj['source'].isin(pages) & adj['target'].isin(pages), :]
    Gs = nx.from_pandas_edgelist(edges)

    largest_cc = sorted(list(nx.connected_components(Gs.to_undirected())), key=len, reverse=True)[0]
    Gs = Gs.subgraph(largest_cc)

    selected_nodes = pd.Series(list(Gs.nodes))
    selected_nodes.to_csv('intermediate/selected-nodes.csv', index=False)

    #
    # Prepare final files
    #

    # graph
    graph = adj.loc[adj['source'].isin(selected_nodes) & adj['target'].isin(selected_nodes), :]
    graph.to_csv('raw/graph.csv', index=False, sep='\t')

    # text
    with gzip.open('intermediate/text-full.csv.gz', 'rt') as f:
        text = pd.read_csv(f)

    text['node_id'] = text['node_id'].str.replace('Q', '').astype(int)
    text = text.loc[text['node_id'].isin(selected_nodes), :]
    text = text.rename({'text': 'content'}, axis=1)

    text.to_csv('raw/texts.csv', index=False, sep='\t')

    # node data (categories)
    node_data = art_cat.loc[art_cat['page_id'].isin(selected_nodes), :] \
        .merge(cats, on='cat_id') \
        [['cat_id', 'page_id']] \
        .drop_duplicates() \
        .assign(dummy=1) \
        .pivot('page_id', 'cat_id', 'dummy') \
        .fillna(0) \
        .astype(int) \
        .reset_index()

    node_data.columns = ['node_id'] + ['cat_' + str(c) for c in list(node_data)[1:]]
    node_data.to_csv('raw/node-data.csv', index=False, sep='\t')
