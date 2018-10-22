from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tfidf import CustomTfidfVectorizer
import pickle
import nltk


class Table:
    def __init__(self, docindex=None, parse_sentences=True, ngram_range=(1, 1),
                 language='portuguese', noun_phrases=False, stemming=False, stop_words=False,
                 verbose=False, max_df=1.0, cut_off=0.0, path=None):

        self.verbose = verbose
        if self.verbose:
            print('Created analyzer table with properties:')
            print('parse_sentences = ' + str(parse_sentences))
            print('ngram_range = ' + str(ngram_range))
            print('language = ' + str(language))
            print('noun_phrases = ' + str(noun_phrases))
            print('stemming = ' + str(stemming))
            print('stop_words = ' + str(stop_words))

        self.path = path
        if self.path is None:
            self.path = 'data/flat_text/'

        self.language = language
        self.ngram_range = ngram_range
        self.noun_phrases = noun_phrases
        self.stemming = stemming
        self.stop_words = stop_words
        self.max_df = max_df
        self.cut_off = cut_off

        self.corpus = []
        self.orig_corpus = []
        self.trained = False

        self.vectorizer = None
        self.bm25vectorizer = None
        self.matrix = None
        self.bm25_matrix = None
        self.vocab = None

        if docindex is not None:
            self.parse_file(docindex, parse_sentences)

    @staticmethod
    def _create_tagger():
        with open('tagger.p', 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def _create_parser():
        grammar = 'NP: {<adj>? <prep>? <n.*>+ <adj>?}'
        return nltk.RegexpParser(grammar)

    @staticmethod
    def _extract_np_from_tree(tree):
        def leaves(tree):
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                yield subtree.leaves()

        phrases = []
        for leaf in leaves(tree):
            if len(leaf) > 1:
                as_string = ''
                for elem in leaf:
                    as_string += elem[0] + ' '
                phrases.append(as_string[:-1])
        return phrases

    @staticmethod
    def _create_vocab(corpus):
        vocab = []
        for doc in corpus:
            for word in tokenize(doc):
                if len(word) > 1:
                    vocab.append(word)
        return set(vocab)

    @staticmethod
    def extract_np(corpus):
        phrases = []
        tagger = Table._create_tagger()
        parser = Table._create_parser()
        for doc in corpus:
            words = tokenize(doc)
            tagged = tagger.tag(words)
            tree = parser.parse(tagged)
            phrases += Table._extract_np_from_tree(tree)
        return phrases

    @staticmethod
    def np_tokenizer(s):
        words = [word for word in tokenize_word(s) if len(word) > 1 and word.isalnum()]
        np = Table.extract_np([s])
        return words + np

    def parse_file(self, filenames, parse_sentences):
        for filename in filenames:
            with open(self.path + filename, 'r', encoding='latin1') as f:
                if parse_sentences:
                    docs = tokenize_sentences(f.read(), language=self.language)
                else:
                    docs = [f.read()]

                self.corpus = self.corpus + docs
        try:
            self.init(self.corpus)
        except ValueError:
            print("ERROR: Failure in parsing corpus")

    def init(self, corpus):
        # learn the vocabulary and frequencies from the training corpus provided
        # keep matrices for cosine similarities
        self.corpus = corpus
        self.orig_corpus = list(self.corpus)
        self.corpus = self.transform(self.corpus)
        self.trained = True

        if self.noun_phrases:
            self.vectorizer = CustomTfidfVectorizer(ngram_range=self.ngram_range, tokenizer=self.np_tokenizer,
                                                    max_df=self.max_df, min_df=self.cut_off)
            self.bm25vectorizer = CustomTfidfVectorizer(bm25=True, ngram_range=self.ngram_range,
                                                        tokenizer=self.np_tokenizer, max_df=self.max_df, min_df=self.cut_off)
        else:
            self.vectorizer = CustomTfidfVectorizer(ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.cut_off)
            self.bm25vectorizer = CustomTfidfVectorizer(bm25=True, ngram_range=self.ngram_range, max_df=self.max_df,
                                                        min_df=self.cut_off)

        self.matrix = self.vectorizer.fit_transform(self.corpus)
        self.bm25_matrix = self.bm25vectorizer.fit_transform(self.corpus)

        self.vocab = self.vectorizer.get_feature_names()
        self.trained = True

    def transform(self, corpus):
        if self.stemming:
            stemmer = nltk.stem.RSLPStemmer()
            for i in range(len(corpus)):
                doc = corpus[i]
                for word in tokenize(doc):
                    new_word = stemmer.stem(word)
                    doc = doc.replace(word, new_word)
                corpus[i] = doc
        return corpus

    def stats(self):
        return {'n_terms': self.n_terms(),
                'n_docs': self.n_docs()}

    def get_corpus(self):
        return self.corpus

    def get_original_corpus(self):
        return self.orig_corpus

    def print_stats(self):
        stats = self.stats()
        print('Unique terms = ' + str(stats['n_terms']))
        print('Number of documents = ' + str(stats['n_docs']))

    def n_docs(self):
        return self.matrix.shape[0]

    def n_terms(self):
        return len(self.vocab)

    def tf_idf_vect(self, doc):
        if self.trained is False:
            raise Exception('The table must be trained before being used')
        # Apply any transformation required by the parameters
        doc = self.transform(doc)
        return self.vectorizer.transform(doc)

    def bm25_vect(self, doc):
        if self.trained is False:
            raise Exception('The table must be trained before being used')

        # Apply any transformation required by the parameters
        doc = self.transform(doc)
        return self.bm25vectorizer.transform(doc)

    def similarity(self, query, bm25=True):
        if bm25:
            query_vect = self.bm25_vect([query])
            result = cosine_similarity(self.bm25_matrix, query_vect)
        else:
            query_vect = self.tf_idf_vect([query])
            result = cosine_similarity(self.matrix, query_vect)
        return result

    def independent_similarity(self, s1, s2, bm25=False):
        if bm25:
            vec1 = self.bm25_vect([s1])
            vec2 = self.bm25_vect([s2])
            result = cosine_similarity(vec1, vec2)
        else:
            vec1 = self.tf_idf_vect([s1])
            vec2 = self.tf_idf_vect([s2])
            result = cosine_similarity(vec1, vec2)
        return result
