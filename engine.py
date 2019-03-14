import re, math
from indexing import IndexBuilder
from nltk.stem.porter import *

class Engine:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.index_builder = IndexBuilder("./data")
        self._inverted_doc_frequency = self.index_builder.idf()
        self._documents = self.index_builder.documents()
        self._terms = self.index_builder.terms()
        self.vector = self.index_builder.vector()
        
  
    def vector_for_query(self, query):
        clean_query = self._clean_query(query)
        query_tf = {}
        for keyword in clean_query:
            if keyword in query_tf:
                query_tf[keyword] += 1
            else:
                query_tf[keyword] = 1
        q_vector = {}
        for doc in self._documents:
            q_vector[doc] = []
            query_tf_idf = {}
            for term in self._terms[doc]:
                if term in clean_query:
                    query_tf_idf[term] = query_tf[term] * self._inverted_doc_frequency[doc][term]
                    q_vector[doc].append(query_tf_idf[term])
                else:
                    q_vector[doc].append(0)
        return q_vector

    def _clean_query(self, query):
        pattern = re.compile('[\W_]+')
        query = pattern.sub(' ', query)
        query = query.split()
        return [self.stemmer.stem(q) for q in query]

    def consine_similarity(self, v1, v2):
        if len(v1) != len(v2):
            return 0
        dot_product = sum([a * b for a, b in zip(v1, v2)])
        v1_mode = math.sqrt(sum( x * x for x in v1))
        v2_mode = math.sqrt(sum( x * x for x in v2))
        if v1_mode == 0 or v2_mode == 0:
            return 0
        return dot_product / (v1_mode * v2_mode)

    def query(self, query):
        v_query = self.vector_for_query(query)
        v_doc = self.vector
        score = {}
        for doc in v_query.keys():
            v1 = v_query[doc]
            v2 = v_doc[doc]
            sim_score = self.consine_similarity(v1, v2) 
            if sim_score > 0:
                score[doc] = sim_score
        ranking_result = sorted(score.items(), key=lambda kv: kv[1], reverse = True)
        return ranking_result