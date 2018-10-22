from eval import *


def main():
    print("-------- Exercise 4 ---------")
    print("- Calculating summaries using the MMR metric as described in Exercise 4")
    ex4()
    print("- Calculating summaries using the baseline of choosing the first five sentences of each text")
    ex4_1()


def ex4():
    files = os.listdir("data/flat_text")
    p, r, f1, map_ = full_evaluate(MMR, files, 5, tab=None, bm25=True, language='portuguese')
    print("P = " + str(p))
    print("R = " + str(r))
    print("F1 = " + str(f1))
    print("MAP = " + str(map_))

def ex4_1():
    files = os.listdir("data/flat_text")
    p, r, f1, map_ = full_evaluate(baseline, files, 5, tab=None, language='portuguese')
    print("P = " + str(p))
    print("R = " + str(r))
    print("F1 = " + str(f1))
    print("MAP = " + str(map_))


if __name__ == '__main__':
    main()

