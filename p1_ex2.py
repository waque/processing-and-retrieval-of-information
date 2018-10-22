from table import *
from utils import *
from eval import *
import os


if __name__ == "__main__":
    path = 'data/flat_text/'
    files = os.listdir("data/flat_text")
    simp_table = Table(files, parse_sentences=True)

    print("-------- Exercise 2 ---------")
    print("- Comparing alternative 1 to alternative 2")
    print("<<< Alternative 1: tf-idf per document (implemented in ex.1) >>>")


    p, r, f1, map_ = full_evaluate(summary, files, 5, tab=None, bm25=False,
                                   language='portuguese', parse_sentences=True)

    evaluate_print("Alternative 1", p, r, f1, map_)

    print("<<< Alternative 2: tf-idf trained with all documents (described in exercise 2) >>>")
    p, r, f1, map_ = full_evaluate(summary, files, 5, tab=simp_table, bm25=False,
                                   language='portuguese', parse_sentences=True)

    evaluate_print("Alternative 2", p, r, f1, map_)
