## This is a source code repository for the 2021 Batchelors thesis "Jēdzientelpu veidošanas metodes latviešu valodā" by Rolands Laucis at the Technical University of Riga (RTU)
### English title: "Word Embedding Methods for the Latvian Language"

Each python script contains a header line explaining its purpose and CLI use examples.
There are also automation .bat files for running model training and evaluation.

The work is done using the python language.
Used libraries and other dependencies are listed in ./dependencies.json

FastText and Word2vec embeddings are trained with the python lib "Gensim".
Ngram2vec and Structure Skip-Gram embeddings are trained with a tool from their original github repositories.

The SpaCy directory contains all source code files and instructions for building and evaluating POS and NER models using custom latvian word embeddings.
