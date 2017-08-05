"""Classify the wikinews set for categories.

This means multilabel classification (a single document often has many different
categories), so I'm using SVM with one vs rest.
"""
import json
import pickle
import os

import click
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from utils import get_category_info, load_data, tokenize


def get_data_and_labels(raw_data, categories):
    """Transform raw data into tfidf vectors and label vector."""
    data = []
    for d in raw_data:
        # filter out the categories we don't care about
        d['categories'] = [c for c in d['categories'] if c in categories]
        if d['categories']:
            data.append(d)
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    raw_labels = [d['categories'] for d in data]
    data = vectorizer.fit_transform([d['text'] for d in data])
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(raw_labels)
    return data, labels, vectorizer, mlb


def classify(data, labels, train_full_model=True):
    """Train and evaluate a classifier for text categories.

    Train and evaluate a classifier with cross-validation, prints the results (using hamming loss
    for scoring, since this is a multi-label classifier). Uses SVM with One vs Rest classifier.

    If train_full_model is true, then also train a model on the complete dataset.
    """
    clf = OneVsRestClassifier(SVC())
    scores = cross_val_score(clf, data, labels, scoring=make_scorer(hamming_loss, greater_is_better=False))
    print(scores)
    print('mean hamming loss:', sum(scores) / len(scores))
    model = None
    if train_full_model:
        model = OneVsRestClassifier(SVC(probability=True)).fit(data, labels)
    return scores, model


@click.command()
@click.option('--input_dir', '-i', help='Input dir with data files, one json object '
                                        'per line, with text and categories fields')
@click.option('--category_dir', '-c', help='Directory containing human-labeled categories, for '
                                          'which categories should be included in the analysis,')
@click.option('--model_output', '-o', type=str, default=None, help='Filename for classifier output')
@click.option('--vectorizor_output', '-v', type=str, default=None, help='Filename for vectorizor output')
@click.option('--label_binarizer_output', '-l', type=str, default=None, help='Filename for label binarizer output')
@click.option('--cat_type', default='all', help='high_level|low_level|all')
def main(input_dir, category_dir, model_output, vectorizor_output, label_binarizer_output, cat_type):
    high_level_cats, low_level_cats = get_category_info(category_dir)
    raw_data = load_data(input_dir)
    cats = high_level_cats if cat_type == 'high_level' else (
        low_level_cats if cat_type == 'low_level' else high_level_cats.union(low_level_cats))
    data, labels, vectorizor, label_binarizer = get_data_and_labels(raw_data, list(cats))
    _, model = classify(data, labels, train_full_model=(model_output is not None))
    if model_output:
        with open(model_output, 'wb') as f:
            pickle.dump(model, f)
    if vectorizor_output:
        with open(vectorizor_output, 'wb') as f:
            pickle.dump(vectorizor, f)
    if label_binarizer_output:
        with open(label_binarizer_output, 'wb') as f:
            pickle.dump(label_binarizer, f)

if __name__ == '__main__':
    main()