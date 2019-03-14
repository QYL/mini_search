from engine import Engine
from flask import Flask
from flask import render_template, request

import re, io, string
from nltk.stem.porter import *

stemmer = PorterStemmer()

app = Flask(__name__)

engine = Engine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    result = None
    query = request.args['query']
    query_terms = query.split() 
    query_terms = [stemmer.stem(t) for t in query_terms]
    result = engine.query(query)
    content = []
    for doc in result:
        document = {}
        doc_name, doc_score = doc
        doc_path = "./data/{0}".format(doc_name)
        doc_content = io.open(doc_path, 'r', encoding="ISO-8859-1").read();
        document["title"] = doc_name
        document["score"] = doc_score
        doc_terms = doc_content.split()
        doc_terms_stemmed = [stemmer.stem(t.translate(str.maketrans('', '', string.punctuation))) for t in doc_terms]
        for term in query_terms:
            positions = [i for i, t in enumerate(doc_terms_stemmed) if t == term]
            for pos in positions:
                doc_terms[pos] = "<font color='red'>{0}</font>".format(doc_terms[pos])    
        document["content"] = " ".join(doc_terms)
        content.append(document)
    return render_template('index.html', content=content)



if __name__ == "__main__":
    app.run(debug = True)
    # workon lab && cd code && python test.py
    # index_builder = IndexBuilder("./data")
    # print(index_builder.df())
    # print(index_builder.tf())
    # print(index_builder.index())
    # print(engine.query("putin plead close"))