## This is a source code repository for the 2021 paper "Evaluation of embedding models on Latvian NLP tasks based on publicly available corpora" by Rolands Laucis and Gints Jēkabsons at the Technical University of Riga (RTU)

Each python script contains a header line explaining its purpose and CLI use examples.
There are also automation .bat files for running model training and evaluation.

The work is done using the python language. You must install the python library dependencies listed in dependencies.json or by running ``pip install -r dependencies.txt`` from the directory.

FastText and Word2vec embeddings are trained with the python lib "Gensim".
Ngram2vec, Structure Skip-Gram and GloVe embeddings are trained with the tools from their original github repositories.

The SpaCy directory contains all source code files and readme.md instructions for building and evaluating POS and NER models using custom latvian word embeddings.
