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

    ranked = pr.rank(prior_calc=PageRanker.prior_N,
                     prestige_calc=PageRanker.prestige_1_level)

    return sorted(ranked, key=ranked.get, reverse=True)[:n]


def main():
    print("-------- Exercise 2 ---------")

    evaluate_print("Pagerank", *full_evaluate(pr_summary, 5, "x", "y"))


if __name__ == '__main__':
    main()
