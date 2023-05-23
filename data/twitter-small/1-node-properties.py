#!/usr/bin/env python3

if __name__ == '__main__':
    import os

    import pandas as pd
    import networkx as nx
    import community

    os.makedirs('derived', exist_ok=True)

    node_data = pd.read_csv('raw/node-data.csv', sep='\t', index_col='node_id')
    edges = pd.read_csv('raw/graph.csv', sep='\t')
    G = nx.from_pandas_edgelist(edges)

    #
    # Self-supervision data for graph model pretraining
    #

    props = {
        'pagerank': nx.pagerank(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'community': community.best_partition(G.to_undirected())
    }

    props = pd.concat([
        pd.Series(val, name=key)
        for key, val in props.items()
    ], axis=1)
    props.index.name = 'node_id'

    #
    # Geographic self-reports, hand-cleaned
    #

    locs = node_data.location.copy()
    locs.name = 'geo'

    locs[locs.str.contains(', NY').fillna(False)] = 'NY'
    locs[locs.str.contains('New York').fillna(False)] = 'NY'
    locs[locs.str.contains('new york').fillna(False)] = 'NY'
    locs[locs.str.contains('NEW YORK').fillna(False)] = 'NY'
    locs[(locs.str.strip() == 'NYC').fillna(False)] = 'NY'
    locs[(locs.str.strip() == 'NYC').fillna(False)] = 'NY'
    locs[(locs.str.strip() == 'nyc').fillna(False)] = 'NY'
    locs[(locs.str.strip() == 'Brooklyn').fillna(False)] = 'NY'

    locs[locs.str.contains(', DC').fillna(False)] = 'DC'
    locs[locs.str.contains(', D.C.').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'Washington DC').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'washington, d.c.').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'Washington D.C.').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'D.C.').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'dc').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'Alexandria, VA').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'Arlington, VA').fillna(False)] = 'DC'
    locs[(locs.str.strip() == 'Washington').fillna(False)] = 'DC'

    locs[locs.str.contains(', CA').fillna(False)] = 'CA'
    locs[locs.str.contains('California').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'LA').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'L.A.').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'Hollywood').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'Hollywood, Los Angeles').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'San Francisco').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'SF').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'Los Angeles').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'los angeles').fillna(False)] = 'CA'
    locs[(locs.str.strip() == 'los angeles, ca').fillna(False)] = 'CA'

    locs[locs.str.lower().str.contains('london').fillna(False)] = 'UK'
    locs[locs.str.contains('United Kingdom').fillna(False)] = 'UK'

    locs[~locs.isin(['DC', 'NY', 'CA', 'UK'])] = 'other'

    props['location_self_report'] = locs
    props.to_csv('derived/node-properties.csv', sep='\t')
