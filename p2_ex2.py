#!/usr/bin/env python3
import itertools

from table import Table
from pageRanker import PageRanker
from utils import tokenize_sentences
from eval import full_evaluate, evaluate_print
import warnings
warnings.filterwarnings('ignore')


def pr_summary(doc, n, prior_fn, weight_fn):
    table = Table()
    table.init(tokenize_sentences(doc))
    pr = PageRanker(table)
    PageRanker.init(table, doc)

    ranked = pr.rank(prior_fn=prior_fn,
                     weight_fn=weight_fn,
                     prior_calc=PageRanker.prior_quotient,
                     prestige_calc=PageRanker.prestige_2_levels)

    return sorted(ranked, key=ranked.get, reverse=True)[:n]


def main():
    print("-------- Exercise 2 ---------")

    prior_fn = (PageRanker.prior_relevance, PageRanker.prior_sentence_ix)
    weight_fn = (PageRanker.weight_shared_noun_phrases, PageRanker.weight_pca, PageRanker.weight_relevance)
    for pfn, wfn in itertools.product(prior_fn, weight_fn):
        print("%s %s" % (pfn, wfn))
        evaluate_print(str((pfn, wfn)),
                       *full_evaluate(pr_summary, 5, pfn, wfn))


if __name__ == '__main__':
    main()
