from table import Table
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def _default(*args, **opt):
    return 1


class PageRanker:

    _sims = None
    _svd = None
    _sim_table = {}
    _np_table = {}
    _v_table = {}

    def __init__(self, tab, sim=0, bm25=True):
        """

        :param tab: The table
        :param sim: Threshold of similarity to consider phrases connected in the graph
        :param bm25: Method to calculate similarity from
        """

        # Optional parameters
        self._sim = sim
        self._bm25 = bm25

        self._table = tab
        self._graph = self._init_graph()

    def _init_graph(self):

        def add(g, key, val):
            if key not in graph:
                g[key] = set()
            g[key].add(val)
            print("hello")
        graph = {}  # Filter out repeated sentences if they exist
        for s1 in self._table.corpus:
            for s2 in self._table.corpus:
                if s1 != s2:
                    if self._table.independent_similarity(s1, s2, bm25=True) >= self._sim:
                        add(graph, s1, s2)
                        add(graph, s2, s1)
        return graph

    def rank(self,
             prior_fn=_default,
             weight_fn=_default,
             prior_calc=_default,
             prestige_calc=_default, iterations=2, rw=0.15):
        """

        :param prior_fn: Function to calculate the prior of each node
        :param weight_fn: Weight function for nodes
        :param prior_calc: Function to calculate of the prior part of the ranking function
        :param prestige_calc: Function to calculate the prestige part of the ranking function
        :param iterations: Number of iterations
        :param rw: Random walk probability
        :return: Ranked sentences
        """
        priors = prior_fn(self._table, bm25=self._bm25)
        ranks = {node: 1 / len(self._graph) for node in self._graph}  # Initial prestige vector
        for i in range(iterations):
            print('IT %d' % i)
            new_ranks = {}
            for node in self._graph:
                val1 = rw * prior_calc(self._graph, priors, node)
                val2 = (1 - rw) * prestige_calc(self._table, self._graph, node, ranks, weight_fn)

                new_ranks[node] = val1 + val2
            ranks = new_ranks

        return ranks

    @staticmethod
    def init(table, corpus):
        PageRanker._sims = table.similarity(" ".join(corpus))
        PageRanker._sim_table = {}
        PageRanker._np_table = {}
        PageRanker._v_table = {}
        PageRanker._svd = TruncatedSVD(n_components=3)
        PageRanker._svd.fit_transform(table.matrix)

    # 1 (right hand side)
    @staticmethod
    def prestige_1_level(table, graph, node, ranks, weight_fn, **opt):
        return sum([ranks[node] / len(graph[node]) for node in graph])

    # 1 (left hand side)
    @staticmethod
    def prior_N(graph, priors, node):
        return 1 / len(graph[node])

    # 2 (left hand site)
    @staticmethod
    def prior_quotient(graph, priors, node):
        return priors[node] / sum([priors[adj] for adj in graph[node]])

    # 2 (right hand side)
    @staticmethod
    def prestige_2_levels(table, graph, pi, ranks, weight_fn, **opt):
        acc = []

        for pj in graph[pi]:

            first_level = ranks[pj] * weight_fn(table, pj, pi, **opt)

            acc_sum = 0
            for pk in graph[pj]:
                acc_sum += weight_fn(table, pj, pk, **opt)

            if acc_sum == 0:
                acc.append(0)
            else:
                acc.append(first_level / acc_sum)

        return sum(acc)

    # 2.1
    @staticmethod
    def prior_sentence_ix(table, **opt):
        corpus = table.corpus
        N = len(corpus)  # Smoothing parameter
        return {corpus[ix]: 1 / (N + ix + 1) for ix in range(len(corpus))}

    # 2.2
    @staticmethod
    def prior_relevance(table, **opt):
        corpus = table.corpus
        PageRanker._sims = table.similarity(" ".join(corpus))

        res = {}
        for ix in range(len(corpus)):
            res[corpus[ix]] = PageRanker._sims[ix]
        return res

    # 2.3
    @staticmethod
    def weight_relevance(table, s1, s2, **opt):
        sim = PageRanker._sim_table.get(s1, None)
        if sim is None:
            sim = table.similarity(s1, **opt)
            PageRanker._sim_table[s1] = sim

        # v2 = able.independent_similarity(s1, s2, bm25=True)
        idx = table.corpus.index(s2)
        return sim[idx]

    # 2.4
    @staticmethod
    def weight_shared_noun_phrases(table, s1, s2, **opt):
        np_s1 = PageRanker._np_table.get(s1, None)
        if np_s1 is None:
            np_s1 = Table.extract_np([s1])
            PageRanker._np_table[s1] = np_s1

        np_s2 = PageRanker._np_table.get(s2, None)
        if np_s2 is None:
            np_s2 = Table.extract_np([s2])
            PageRanker._np_table[s2] = np_s2
        return len([np for np in np_s1 if np in np_s2])


    @staticmethod
    def weight_pca(table, s1, s2):
        sim = PageRanker._v_table.get(s1, None)
        if sim is None:
            v1 = PageRanker._svd.transform(table.tf_idf_vect([s1]))
            all = PageRanker._svd.transform(table.matrix)
            sim = cosine_similarity(v1, all)[0]
        PageRanker._v_table[s1] = sim

        idx = table.corpus.index(s2)
        res = sim[idx]
        return res
