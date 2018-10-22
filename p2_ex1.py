from pageRanker import PageRanker
from table import *
from utils import *
from utils2 import *
from eval import *


def main():
    print("-------- Exercise 1 ---------")
    print("- Using PageRank approach to summarize an english textual document")
    print("<<< Summary of data/ex1.txt is >>>")
    directory = 'data/flat_text/'
    table = Table(['St-ce94ab10-a.txt'], parse_sentences=True, language='english', path=directory)
    with open(directory + 'St-ce94ab10-a.txt', 'r', encoding='latin1') as file:
        doc = file.read()
        table = Table()
        table.init(tokenize_sentences(doc))

        pr = PageRanker(table, sim=0.2)
        PageRanker.init_sims(table, doc)
        ranked = pr.rank(prior_calc=PageRanker.prior_N,
                         prestige_calc=PageRanker.prestige_1_level)

        result = [s.lower() for s in sorted(ranked, key=ranked.get, reverse=True)[:5]]
        print(result)

        #dic = evall(result, 'Ext-ce94ab10-a.txt')
        #p = precision(dic['tp'], dic['r'] - dic['tp'])
        #r = recall(dic['tp'], dic['len_collection_sum'] - dic['tp'])
        #f1_score = F1_score(p, r)
        #print("\n")
        #print("precision: " + str(p))
        #print("recall: " + str(r))
        #print("F1 score: " + str(f1_score))


if __name__ == '__main__':
    main()
