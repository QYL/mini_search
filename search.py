from engine import Engine
from flask import Flask
from flask import render_template, request
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import re, io, string, os
from nltk.stem.porter import *


UPLOAD_FOLDER = './data'

stemmer = PorterStemmer()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

engine = Engine()

@app.route('/')
def index():
    content = {}
    return render_template('index.html', content=content)

@app.route('/search')
def search():
    result = None
    query = request.args['query']
    query_terms = query.split() 
    query_terms = [stemmer.stem(t) for t in query_terms]
    result = engine.query(query)
    content = {}
    content["documents"] = []
    content["query"] = request.args['query']
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
        content["documents"].append(document)
    return render_template('index.html', content=content)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file found.')
            return render_template('upload_status.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file.')
            return render_template('upload_status.html')
        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File Uploaded!')
        engine.build()
        return render_template('upload_status.html')
    return render_template('upload.html')

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug = True)