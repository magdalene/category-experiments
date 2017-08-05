This project contains some scripts for simple "proof of concept" experiments tagging text with categories using Wikinews (crowd-sourced news) data. It is not expected that this project can be used as is, but instead provides some examples to expand on.

### Corpus

I've used the database backup dump of Wikinews, downloaded from the [wikimedia dumps page](https://dumps.wikimedia.org/).

The Wikinews dump contains crowd-sourced news articles, and each article has associated categories, which can provide labeled training data for supervised learning, or more straightforward evaluation for unsupervised approaches.

Wikinews has 13 top-level categories (e.g., Politics and Conflicts, Science and Technology), and many lower-level categories.

### Preprocessing

* [Wikiextractor](https://github.com/attardi/wikiextractor) tool -- to extract and clean texts from the dump, called with `--json -ns Category`, for json output, and preserving category information
* `mark_categories.py` -- move the categories from the body of the text, to a separate field in the json object for each document
* [List of countries](https://github.com/mledoze/countries) -- to filter out and exclude country categories
* for each top-level category, generate a csv with rows that contain low level category name,count,yes|no|place (TODO: write a script for this)
** low level category name -- string like "Internet" (occuring in Science_and_technology.csv)
** count -- number of documents in which this low level category co-occurs with the top-level category
** yes|no|place -- only categories marked "yes" is included in training or evaluation (this is a human judgment, about whether the category actually belongs to the top-level category, and also whether it's an interesting category for the particular use-case)
* Each learning script (`cluster.py`, `classify.py`) uses [nltk](https://github.com/nltk/nltk) for tokenization and stemming, and then uses tf-idf vectors to represent the text


### Category selection

For the experiments, I excluded all named entity categories, including geography-based (continents, countries, cities, bodies of water, etc.), and people (world leaders). Otherwise I kept most categories, as long as they subjectively seemed to belong to the high-level category. Some low-level categories (~5) are associated with multiple top-level categories (e.g., Transportation belongs both to Economy and business, and Science and technology).

### Unsupervised (clustering) approach

`cluster.py` creates clusters of documents using k-means, with the hypothesis that with sufficient number of clusters, the clusters will correspond to a human-understandable category.

`get_cluster_keywords.py` generates a csv with cluster id and a list of keywords for each cluster, for human annotation, in order to assign each cluster to a top-level category. This allows for "realistic" unsupervised evaluation.

`eval_cluster_best_match.py` calculates accuracy assuming perfect human annotator performance, giving an upper-bound on performance (taking the most common category for each cluster, as its label).

### Supervised (classification) approach

`classify.py` trains a multilabel classifier (sklearn OneVsRestClassifier, with SVM) to predict the list of categories for each text. Uses cross-validation and reports hamming loss scores. Optionally computes a final model trained on the full dataset with the option to predict probabilities of each class, and outputs the pickled model, vectorizor, and label vectorizor for future use.

### Realistic test set

I generated a realistic test set, by copying text from various news websites, and manually labeling with a high-level category. The test set consisted of 57 documents, 13 from Politics and Conflicts (the most frequent top-level category), and the remaining texts approximately evenly distributed between the other categories, except excluding Obituaries entirely. This test set is interesting, because it's qualitatively different than the Wikinews articles, sometimes in writing style, and usually in length (the Wikinews stories are typically much shorter).

### Performance

Ultimately (and unsurprisingly!) the supervised approach works best. Using a classifier trained on only low-level categories, and extrapolating high-level categories from low-level categories, taking the top 2 low-level categories (by probability), 94.7% of the extrapolated high level categories contain the correct category (on the "realistic" test set). This performance is possibly sufficient for some tasks, and can certainly be improved by supplementing the training set with more data for the sparsely populated categories.

While keywords from the clusters appear to usually "make sense" (that is, correspond cleanly to a single top-level category), performance on the realistic test set with human-annotated is very poor (accuracy .38), and best-case accuracy on the original dataset (based on most common category for the cluster) is .68.

Some observations:

* The classifier model predicted no classes for any of the "realistic" test documents. This might be solved by normalizing the tf-idf vectors, but the probability prediction is sufficient for many uses.
* The low-level "Missing person" category appeared erroneously all over the place in the results for the "realistic" test; some investigation is in order to understand why this happens, and ameliorate the issue (if using low-level categories)
* The three "completely wrong" predictions were on articles about sports (in particular, golf, tennis, and soccer) where the data from Wikinews is sparse. For applications where sports are important, more data would be necessary.

### License

Everything here is licensed under the new BSD license. Please see [LICENSE.md](LICENSE.md) for details.