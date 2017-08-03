"""Modified extract_key_phrases from textrank, to return ordered (instead of unordered) keywords."""

from textrank import filter_for_tags, normalize, unique_everseen, build_graph
import networkx as nx
import nltk

### This is all copied from textrank...
def extract_key_phrases(text):
    """Return a set of key phrases.
    :param text: A string.
    """
    # tokenize the text using nltk
    word_tokens = nltk.word_tokenize(text)

    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph(word_set_list)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    one_third = len(word_set_list) // 3
    keyphrases = keyphrases[0:one_third + 1]

    ########### Here begin the modifications, completely rewritten to preserve rank #########
    keyphrase_to_idx = {keyphrase: i for i, keyphrase in enumerate(keyphrases)}

    candidate_bigrams = set(['{} {}'.format(i, j) for i, j in zip(textlist[:-1], textlist[1:])])
    used_unigrams = set()
    modified_key_phrases = {}

    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together

    # NOTE: This problem exists in the original code also, but the problem here is that we're prioritizing
    # making a bigram based on first appearance in the text, rather than pagerank scores. In practice the outcome
    # seems ok though.
    for bigram in candidate_bigrams:
        first, second = bigram.split()
        if first in keyphrases and second in keyphrases and first not in used_unigrams and second not in used_unigrams:
            modified_key_phrases[min(keyphrase_to_idx[first], keyphrase_to_idx[second])] = bigram
            used_unigrams.add(first)
            used_unigrams.add(second)

    for i, keyphrase in enumerate(keyphrases):
        if keyphrase not in used_unigrams:
            modified_key_phrases[i] = keyphrase

    return sorted([x[1] for x in modified_key_phrases.items()])




