from eval import *


def main():
    print("-------- Exercise 3 ---------")
    print("<<< Comparing different enhancements of the tf-idf approach in Exercise 1 >>>")

    print("<<< Calculating summaries using unigrams to bigrams and noun phrases >>>")
    ex3_1()
    print("<<< Calculating summaries using bm25 metric >>>")
    ex3_2()
    print("<<< Calculating summaries using our approach of imposing a max document frequency for considered words >>>")
    our_approach_1()
    print("<<< Calculating summaries using our approach of applying stemming to all words before considering them >>>")
    our_approach_2()


def ex3_1():
    files = os.listdir("data/flat_text")
    p, r, f1, map_ = full_evaluate(summary, files, 5, tab=None, bm25=False,
                                   language='portuguese', ngram_range=(1, 2), noun_phrases=True)
    print("P = " + str(p))
    print("R = " + str(r))
    print("F1 = " + str(f1))
    print("MAP = " + str(map_))


def ex3_2():
    files = os.listdir("data/flat_text")
    p, r, f1, map_ = full_evaluate(summary, files, 5, tab=None, bm25=True,
                                   language='portuguese', ngram_range=(1, 1), noun_phrases=False)
    print("P = " + str(p))
    print("R = " + str(r))
    print("F1 = " + str(f1))
    print("MAP = " + str(map_))


def our_approach_1():
    files = os.listdir("data/flat_text")
    p, r, f1, map_ = full_evaluate(summary, files, 5, tab=None, bm25=True,
                                   language='portuguese', ngram_range=(1, 1), noun_phrases=False,
                                   stemming=False, max_df=0.8)
    print("P = " + str(p))
    print("R = " + str(r))
    print("F1 = " + str(f1))
    print("MAP = " + str(map_))

def our_approach_2():
    files = os.listdir("data/flat_text")
    p, r, f1, map_ = full_evaluate(summary, files, 5, tab=None, bm25=False,
                                   language='portuguese', ngram_range=(1, 1), noun_phrases=False,
                                   stemming=True)
    print("P = " + str(p))
    print("R = " + str(r))
    print("F1 = " + str(f1))
    print("MAP = " + str(map_))


if __name__ == '__main__':
    main()
