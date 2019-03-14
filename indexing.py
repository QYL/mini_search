import re, io, math, json
from os import listdir
from os.path import isfile, join
from nltk.stem.porter import *

ignore_list = [".DS_Store", ]

class IndexBuilder:
    
    def __init__(self, doc_path):

        self.stemmer = PorterStemmer()
        self.documents = [f for f in listdir(doc_path) if isfile(join(doc_path, f)) and f not in ignore_list]
        with open('english.stop') as stopwords:
            self.stopwords = stopwords.read().split("\n")
        self.terms = self._create_terms()

        self.term_frequency = {}
        self.doc_frequency = {}
        self.inverted_doc_frequency = {}
        
    # {'D0650.M.250.E.J': ['jimmi', 'carter', '39th', 'presid', ...}
    def _create_terms(self):
        terms = {}
        for doc in self.documents:
            doc_path = "./data/{0}".format(doc)
            pattern = re.compile('[\W_]+')
            terms[doc] = io.open(doc_path, 'r', encoding="ISO-8859-1").read().lower();
            terms[doc] = pattern.sub(' ',terms[doc])
            terms[doc] = terms[doc].split()
            terms[doc] = [self.stemmer.stem(term) for term in terms[doc] if term not in self.stopwords]
        return terms

    # Build the positional index for a doc 
    # return 'jimmi': [0], 'carter': [1, 22, 24, 35, 98, 99, 130, 147], '39th': [2], ...}
    def _positional_index_for_doc(self, doc):
        positional_index = {}
        for index, term in enumerate(self.terms[doc]):
            if term in positional_index.keys():
                positional_index[term].append(index)
            else:
                positional_index[term] = [index]
        return positional_index
    
    # return {'D0650.M.250.E.J': {'jimmi': [0], 'carter': [1, 22, 24, 35, 98, 99, 130, 147], ...}
    def _doc_index(self):
        doc_map = {}
        for file_name in self.terms.keys():
            doc_map[file_name] = self._positional_index_for_doc(file_name)
        return doc_map

    # Build inverted index
    # return {'breach': {'D0634.M.250.G.H': [112], 'D0634.M.250.G.F': [125], 'D0634.M.250.G.I': [80]}, ...}
    def inverted_index(self):
        inverted_index = {}
        doc_index = self._doc_index()
        for file_name in doc_index.keys():
            self.term_frequency[file_name] = {}
            for term in doc_index[file_name].keys():
                self.term_frequency[file_name][term] = len(doc_index[file_name][term])
                if term in self.doc_frequency.keys():
                    self.doc_frequency[term] += 1
                else:
                    self.doc_frequency[term] = 1 
                if term in inverted_index.keys():
                    if file_name in inverted_index[term].keys():
                        inverted_index[term][file_name].append(doc_index[file_name][term][:])
                    else:
                        inverted_index[term][file_name] = doc_index[file_name][term]
                else:
                    inverted_index[term] = {file_name: doc_index[file_name][term]}
        with open('./inverted_index.json', 'w') as json_file:
            json.dump(inverted_index, json_file)
        return inverted_index