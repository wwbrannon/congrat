#!/usr/bin/env python3

'''
Fetch and format the pubmed-diabetes dataset via the Pubmed API
'''

import os
import io
import time
import random
import zipfile
import logging
import xml.etree.ElementTree as et

import pandas as pd
import requests as rq


logger = logging.getLogger(__name__)


class Entry:
    # 1) see https://www.ncbi.nlm.nih.gov/pmc/tools/get-metadata/
    # 2) see also https://www.ncbi.nlm.nih.gov/pmc/tools/cites-citedby/, but
    #    the efetch endpoint seems to have the PMIDs of cited articles already
    # 3) see https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html
    #    for an English description of this XML schema + the DTD
    # 4) no authentication needed, thanks NIH
    _base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='

    def __init__(self, pmid, cache_dir=None):
        super().__init__()

        if isinstance(pmid, str):
            pmid = int(pmid)

        self.data = {'node_id': pmid}
        self.texts, self.references = [], []
        self.api_response = None
        self.cache_dir = cache_dir

        self.is_populated = False

    @property
    def pmid(self):
        return self.data['node_id']

    @property
    def cache_path(self):
        if self.cache_dir is not None:
            return os.path.join(self.cache_dir, str(self.pmid) + '.xml')
        else:
            return None

    @property
    def is_cached(self):
        return self.cache_path is not None and os.path.exists(self.cache_path)

    def ensure_api_response(self):
        # Use the cached version if we have it
        if self.is_cached:
            with open(self.cache_path, 'rt') as f:
                self.api_response = f.read().strip()

            return self

        # Get it if we don't
        resp = rq.get(self._base_url + str(self.pmid))
        self.api_response = resp.text

        # Cache it if we got it
        if self.cache_path is not None:
            with open(self.cache_path, 'wt') as f:
                f.write(self.api_response.strip())

        return self

    def populate(self):
        self.ensure_api_response()

        root = et.fromstring(self.api_response)
        assert root.tag == 'PubmedArticleSet'
        assert len(root) == 1

        obj = root[0]
        assert len(obj) == 2
        assert {c.tag for c in obj} == {'MedlineCitation', 'PubmedData'}

        citation = obj.find('MedlineCitation')
        assert int(citation.find('PMID').text) == self.pmid

        article = citation.find('Article')
        journal = article.find('Journal')
        abstract = article.find('Abstract')

        date = article.find('ArticleDate')
        if date is None:
            date = citation.find('DateCompleted')

        pubmed_data = obj.find('PubmedData')
        assert 'PublicationStatus' in [n.tag for n in pubmed_data]
        assert 'ArticleIdList' in [n.tag for n in pubmed_data]

        ## Texts
        self.texts += [{
            'kind': 'title',
            'order': 0,
            'content': article.find('ArticleTitle').text,
        }]

        for i, para in enumerate(abstract):
            if 'Label' in para.attrib.keys():
                txt = para.attrib['Label'] + '\n'
            else:
                txt = ''

            txt += para.text if para.text is not None else ''

            self.texts += [{
                'kind': 'abstract',
                'order': i,
                'content': txt,
            }]

        ## Article data
        self.data['publication_status'] = pubmed_data.find('PublicationStatus').text

        try:
            self.data['journal_issn'] = journal.find('ISSN').text
        except AttributeError as exc:  # NoneType has no attribute 'text'
            self.data['journal_issn'] = None

        try:
            self.data['journal_name'] = journal.find('Title').text
        except AttributeError as exc:  # NoneType has no attribute 'text'
            self.data['journal_name'] = None

        self.data['date'] = '-'.join([
            date.find('Year').text,
            date.find('Month').text,
            date.find('Day').text,
        ])

        for aid in pubmed_data.find('ArticleIdList'):
            id_type = aid.attrib['IdType']
            self.data['article_id_' + id_type] = aid.text

        # these are "MeSH publication types", see
        # https://www.nlm.nih.gov/mesh/2019/download/NewPubTypes2019.pdf
        # for what the alphanumeric codes mean
        for ptl in article.find('PublicationTypeList'):
            col = 'publication_type_' + ptl.attrib['UI']
            self.data[col] = True

        ## References
        refs = pubmed_data.find('ReferenceList')
        if refs is not None:
            for ref in refs:
                ids = ref.find('ArticleIdList')
                if ids is None:
                    continue

                for ent in ids:
                    if ent.attrib['IdType'] == 'pubmed':
                        self.references += [int(ent.text)]
                        continue

        self.is_populated = True
        return self


class EntrySet:
    def __init__(self, pmids, cache_dir=None):
        super().__init__()

        self.pmids = pmids
        self.cache_dir = cache_dir

        self.entries = [
            Entry(pmid, cache_dir=cache_dir)
            for pmid in self.pmids
        ]

    @property
    def is_populated(self):
        return all(e.is_populated for e in self.entries)

    def populate(self):
        for ent in self.entries:
            if not ent.is_populated:
                if not ent.is_cached:
                    # let's not get rate limited or banned or something
                    if random.random() < 0.05:
                        sleep_time = 6 + random.gauss(0, 1)
                    else:
                        sleep_time = 2 * random.random()
                    logger.debug('Sleeping %s seconds', sleep_time)
                    time.sleep(sleep_time)

                logger.info('Populating %s', ent.pmid)

                try:
                    ent.populate()
                except Exception as exc:
                    logger.exception('Error on populating %s', ent.pmid)

        return self

    def to_pandas(self):
        assert self.is_populated

        # Node data
        node_data = [e.data for e in self.entries]
        node_data = pd.DataFrame.from_dict(node_data, orient='columns')
        for col in node_data.columns:
            if col.startswith('publication_type_'):
                node_data[col] = node_data[col].fillna(False)

            if node_data[col].dtype.name == 'bool':
                node_data[col] = node_data[col].astype(int)

        # Node texts
        texts = pd.DataFrame.from_dict([
            dict(node_id=ent.pmid, **text)
            for ent in self.entries
            for text in ent.texts
        ], orient='columns')

        # Graph edges
        graph = pd.DataFrame.from_dict([
            {'source': ent.pmid, 'target': tgt}
            for ent in self.entries
            for tgt in ent.references
        ], orient='columns')

        # Filter
        graph = graph.loc[graph['source'].isin(node_data['node_id']), :]
        graph = graph.loc[graph['target'].isin(node_data['node_id']), :]

        return {
            'graph': graph,
            'texts': texts,
            'node-data': node_data,
        }


if __name__ == '__main__':
    NODE_PATH = 'pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab'
    GRAPH_PATH = 'pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab'

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
    )

    os.makedirs('raw/', exist_ok=True)
    os.makedirs('raw/cache/', exist_ok=True)

    #
    # Nodes for fetch
    #

    with zipfile.ZipFile('orig/pubmed-diabetes.zip') as zf:
        with io.TextIOWrapper(zf.open(NODE_PATH), encoding='utf-8') as f:
            nodes = pd.read_csv(f, sep='\t', skiprows=1, low_memory=False)

        with io.TextIOWrapper(zf.open(GRAPH_PATH), encoding='utf-8') as f:
            orig = pd.read_csv(
                f, sep='\t', skiprows=2,
                names=['c1', 'source', 'c3', 'target']
            )

    nodes = nodes[nodes.columns[0:2].tolist()]
    nodes.columns = ['node_id', 'label']  # node_id == Pubmed's pmid
    nodes['label'] = nodes['label'].str.replace('label=', '').astype(int)
    nodes = nodes.sample(frac=1)  # fetch in random order to help debugging

    orig = orig.drop(['c1', 'c3'], axis=1)
    orig['source'] = orig['source'].str.replace('paper:', '').astype(int)
    orig['target'] = orig['target'].str.replace('paper:', '').astype(int)

    # this article is no longer on pubmed - retracted or something?
    nodes = nodes.loc[nodes['node_id'] != 17874530, :]

    mask = (
        (orig['source'] != 17874530) &
        (orig['target'] != 17874530)
    )
    orig = orig.loc[mask, :]

    #
    # Fetch (or load from cache) and process these articles
    #

    files = EntrySet(
        nodes['node_id'].tolist(),
        cache_dir='raw/cache/'
    ).populate().to_pandas()

    #
    # Post-hoc processing
    #

    files['node-data'] = files['node-data'].merge(
        nodes, how='left', on='node_id'  # keep the human annotations
    )

    orig['orig'] = 1
    files['graph']['new'] = 1
    files['graph'] = files['graph'].merge(orig, how='outer') \
        .fillna(0) \
        .astype(int)

    for k, v in files.items():
        v.to_csv('raw/{0}.csv'.format(k), index=False, sep='\t')
