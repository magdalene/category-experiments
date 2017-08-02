from collections import Counter
import json
import os

import click

def get_category_info(categories_dir):
    """Load category info from human-labeled files."""
    high_level_cats = set()
    low_level_cats = set()
    for filename in os.listdir(categories_dir):
        with open(os.path.join(categories_dir, filename), 'rt', encoding='utf-8') as f:
            catname = filename.replace('_', ' ').replace('.csv', '')
            high_level_cats.add(catname)
            for line in f:
                line_split = line.strip().split(',')
                if line_split[-1].startswith('yes'):
                    low_level_cats.add(line_split[0])
    return high_level_cats, low_level_cats

def get_cluster_performance(cluster_path, high_level_cats, low_level_cats):
    """Get the performance of a cluster, for high-level cat and low-level cat."""
    high_level_counter = Counter()
    low_level_counter = Counter()
    doc_count = 0
    with open(cluster_path, 'rt', encoding='utf-8') as f:
        for line in f:
            doc_count += 1
            data = json.loads(line)
            for category in data['categories']:
                if category in high_level_cats:
                    high_level_counter[category] += 1
                if category in low_level_cats:
                    low_level_counter[category] += 1
    high_level_correct = high_level_counter.most_common(1)[0][1] if len(high_level_counter) else 0
    low_level_correct = low_level_counter.most_common(1)[0][1] if len(low_level_counter) else 0
    return {'high_level_correct': high_level_correct, 'low_level_correct': low_level_correct, 'doc_count': doc_count}


@click.command()
@click.option('--categories_dir', '-c', help='Directory containing human-labeled categories, for '
                                             'which categories should be included in the analysis')
@click.option('--cluster_dir', '-i', help='Directory with the cluster raw documents created by cluster.py')
def main(categories_dir, cluster_dir):
    high_level_cats, low_level_cats = get_category_info(categories_dir)
    cluster_perfs = []
    for filename in os.listdir(cluster_dir):
        cluster_perfs.append(
            get_cluster_performance(
                os.path.join(cluster_dir, filename),
                high_level_cats, low_level_cats))
    doc_count = sum([cluster_perf['doc_count'] for cluster_perf in cluster_perfs])
    mean_cluster_high_level_accuracy = sum([cluster_perf['high_level_correct']/cluster_perf['doc_count']
                                            for cluster_perf in cluster_perfs]) / len(cluster_perfs)
    mean_high_level_accuracy = sum([cluster_perf['high_level_correct']
                                    for cluster_perf in cluster_perfs]) / doc_count
    mean_cluster_low_level_accuracy = sum([cluster_perf['low_level_correct']/cluster_perf['doc_count']
                                            for cluster_perf in cluster_perfs]) / len(cluster_perfs)
    mean_low_level_accuracy = sum([cluster_perf['low_level_correct']
                                    for cluster_perf in cluster_perfs]) / doc_count
    print('Mean cluster accuracy: high level: {}, low level: {}'.format(
        mean_cluster_high_level_accuracy, mean_cluster_low_level_accuracy))
    print('Mean (total) accuracy: high level: {}, low_level: {}'.format(
        mean_high_level_accuracy, mean_low_level_accuracy))


if __name__ == '__main__':
    main()