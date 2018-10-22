import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


def _add_scalar_sp(sparse,column):
    addition = sp.lil_matrix(sparse.shape)
    sparse_coo = sparse.tocoo()
    for i,j,v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
        addition[i,j] = v + column[i,0]
    return addition.tocsr()


def _divide_scalar_sp(sparse,column):
    div = sp.lil_matrix(sparse.shape)
    sparse_coo = sparse.tocoo()
    column = column.toarray()
    for i, j, v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
        div[i, j] = v / column[i, 0]
    return div.tocsr()


class CustomTfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, norm='l2', use_idf=True, bm25=False):
        self.norm = norm
        self.use_idf = use_idf

        self.bm25 = bm25
        self.k1 = 1.2
        self.b = 0.75
        self._avgdl = 0

    def fit(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        if self.use_idf:
            n_samples, n_features = X.shape

            df = _document_frequency(X)

            if self.bm25:            
                bm25idf = np.log((n_samples - df + 0.5) / (df + 0.5))
                self._avgdl = np.average(X.sum(1))
                self._idf_diag = sp.spdiags(bm25idf, diags=0, m=n_features,
                                            n=n_features, format='csr')

            # Regular idf
            else:
                idf = np.log(float(n_samples) / df)
                self._idf_diag = sp.spdiags(idf,
                    diags=0, m=n_features, n=n_features, format='csr')

        return self

    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.csr_matrix(X, copy=copy)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape
        if self.bm25:
            # reshape is only necessary for certain edge cases
            D = (X.sum(1) / self._avgdl).reshape((n_samples, 1))
            D = ((1 - self.b) + self.b * D) * self.k1
            D_X = _add_scalar_sp(X,D)
            np.divide(X.data * (self.k1 + 1), D_X.data, X.data)
        else:
            #normalized tf uses max term freq of
            F_max = X.max(1)
            X = _divide_scalar_sp(X, F_max)

        if self.use_idf:
            if not hasattr(self, "_idf_diag"):
                raise ValueError("idf vector not fitted")
            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X

    @property
    def idf_(self):
        if hasattr(self, "_idf_diag"):
            return np.ravel(self._idf_diag.sum(axis=0))
        else:
            return None


class CustomTfidfVectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, bm25=False):
        super(CustomTfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = CustomTfidfTransformer(norm=norm, use_idf=use_idf,
                                       bm25=bm25)

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    def fit(self, raw_documents, y=None):
        X = super(CustomTfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        X = super(CustomTfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X,y)
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        X = super(CustomTfidfVectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)
