## This is a source code repository for the 2021 paper "Evaluation of embedding models on Latvian NLP tasks based on publicly available corpora" by Rolands Laucis and Gints Jēkabsons at the Technical University of Riga (RTU)

Each python script contains a header line explaining its purpose and CLI use examples.
There are also automation .bat files for running model training and evaluation.

The work is done using the python language. You must install the python library dependencies listed in dependencies.json or by running ``pip install -r dependencies.txt`` from the directory.

FastText and Word2vec embeddings are trained with the python lib "Gensim".
Ngram2vec, Structure Skip-Gram and GloVe embeddings are trained with the tools from their original github repositories.

The SpaCy directory contains all source code files and readme.md instructions for building and evaluating POS and NER models using custom latvian word embeddings.

There is a folder hierarchy right outside of this repository, that is used in the relative paths in all scripts. It is a bit long to explain, but mainly there are these folders ``datasets``,``Corpora``,``Cleaned_Corpora``,``Models``,``Results``, and the folder names of the various tool repositories. Check the scripts to see what folders need to be created under these.

## Script execution order outline

![alt text](https://github.com/Rolands-Laucis/Word_Embedding_Methods_for_the_Latvian_Language/blob/master/Darba%20ieguld%C4%ABjuma%20diagramma.png)

(Red steps)
Firstly the corpus needs to be built, this is easy with the BuildCorpora.bat file. Check what it does.

(Cyan steps)
Then you build the Ngram2vec and GloVe embeddings separately (because i couldnt get it done with the .bat file on windows). They have their .sh files in this repo and you can copy them over to the tool's cloned repo folder and run. Check them for file paths and other settings. NB! ``glove_train.sh`` has not been tested, since the original used file was lost due to an unfortunate server termination before this dokumentation could be made.
Then you build the word2vec, fastText and SSG embeddings and evaluate all method embeddings on analogies with ``runAll.bat``

(Green and Blue steps)
Head over to the SpaCy folder and read the readme.md instructions there for generating POS and NER models and evaluation.
