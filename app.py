import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import backend as test
import time

app = Flask(__name__)

result = []
documents = {}

# mengambil isi dokumen berdasar index yang sudah didapatkan. pengambilan dokumen dari korpus.txt
def retrieval(result):
    global documents
    documents.clear()
    if (result):
        documents = {k: v for d in result for k, v in d.items()}
    else:
        documents = {}
        
    print(documents)
    return documents

@app.route('/')
def dictionary():
    return render_template('home.html')

@app.route("/query", methods=['POST'])
def upload():
    global documents, result

    start = time.time()
    query = request.form['query']

    # ngehapus isi result biar pencarian sebelumnya tidak keikut
    result.clear()
    result = test.testingDataUji(query, test.token)

    # ngehapus isi documents biar dokumen yang diretrieval tidak keikut
    documents.clear()
    documents = retrieval(result)

    end = time.time()
    times = end - start
    print(times)
    return render_template('result.html', query = query, aspek = documents, jlh_resep = len(documents))

if __name__ == '__main__':
    app.run()