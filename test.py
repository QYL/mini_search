from indexing import IndexBuilder
from flask import Flask
from flask import jsonify

app = Flask(__name__)

index_builder = IndexBuilder("./data")

@app.route('/df')
def df():
    return jsonify(index_builder.df())

@app.route('/idf')
def idf():
    return jsonify(index_builder.idf())

@app.route('/tf')
def tf():
    return jsonify(index_builder.tf())

@app.route('/vector')
def vector():
    return jsonify(index_builder.vector())

@app.route('/index')
def index():
    return jsonify(index_builder.index())

if __name__ == "__main__":
    app.run(debug = True)
    # workon lab && cd code && python test.py
    # index_builder = IndexBuilder("./data")
    # print(index_builder.df())
    # print(index_builder.tf())
    # print(index_builder.index())