# Datasets

We need to fetch several datasets to make this work:
* T-REx: the dataset of articles from aclanthology.org/L18-1544; contains the
  articles ID'd by their wikidata 'Q____' entities, their abstracts, and other
  wikidata entities linked to in those abstracts.
* SameAs: wikidata entity <=> page name and wikidata entity <=> category name
* Page name <=> category name
* Category hierarchy keyed by category name

The idea is: we get a set of texts (abstracts and page names) and a link graph
from T-REx, but it's too big to use. We want to filter down to only a few
categories. To do this, we need the dbpedia datasets that define the categories,
the map from them to Q entities, and their hierarchy.

Fetch the files:
```
# T-REx data
wget -O TREx.zip https://figshare.com/ndownloader/files/8760241

# (article or category) <=> Q entity: https://databus.dbpedia.org/dbpedia/wikidata/sameas-all-wikis
wget https://databus.dbpedia.org/dbpedia/wikidata/sameas-all-wikis/2022.09.01/sameas-all-wikis.ttl.bz2

# Category data: https://databus.dbpedia.org/dbpedia/generic/categories
wget https://databus.dbpedia.org/dbpedia/generic/categories/2022.09.01/categories_lang=en_skos.ttl.bz2
wget https://databus.dbpedia.org/dbpedia/generic/categories/2022.09.01/categories_lang=en_labels.ttl.bz2
wget https://databus.dbpedia.org/dbpedia/generic/categories/2022.09.01/categories_lang=en_articles.ttl.bz2
```
