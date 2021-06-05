::This script converts all word embeddings to spacy format vectors for use in the training stage of POS and NER models using spacy. Was not used in the thesis, this is just for clarification, since this was done manually.

::set up arguments
set win=%1
set v_size=%2

cd ..
::first fasttext and word2vec Gensim .wordvector formats need to be translated to word2vec textual .txt format
python ModelTypeTransform.py --input_file_type wordvectors --input_file ..\Models\Word2vec_model\word2vec_%win%_%v_size%_sg.wordvectors --output_file_type txt --output_file ..\Models\Word2vec_model\word2vec_%win%_%v_size%_sg.txt
python ModelTypeTransform.py --input_file_type wordvectors --input_file ..\Models\FastText_model\fasttext_%win%_%v_size%_sg.wordvectors --output_file_type txt --output_file ..\Models\FastText_model\fasttext_%win%_%v_size%_sg.txt

cd spacy
::then all the word embeddings are converted to spacy format and saved for use in training
python -m spacy init vectors lv ../../Models/FastText_model/fasttext_%win%_%v_size%_sg.txt ../../Models/Spacy_tagger/fasttext_%win%_%v_size%_sg_vectors --name lv-ft-%win%-%v_size%
python -m spacy init vectors lv ../../Models/Word2vec_model/word2vec_%win%_%v_size%_sg.txt ../../Models/Spacy_tagger/word2vec_%win%_%v_size%_sg_vectors --name lv-ssg-%win%-%v_size%
python -m spacy init vectors lv ../../Models/SSG_model/ssg_%win%_%v_size%_sg.txt ../../Models/Spacy_tagger/ssg_%win%_%v_size%_sg_vectors --name lv-ssg-%win%-%v_size%
python -m spacy init vectors lv ..\..\ngram2vec-master\outputs\combined_clean_corpus\ngram_ngram\sgns\ng2v_%win%_%v_size%_sg.output ../../Models/Spacy_tagger/ng2v_%win%_%v_size%_sg_vectors --name lv-ng2v-%win%-%v_size%
echo NB! For glove you have to increment by 1 the first line first number of the .txt, so that spacy can convert it. Because array index issues
python -m spacy init vectors lv ../../Models/Glove_model/glove_%win%_%v_size%.txt ../../Models/Spacy_tagger/glove_%win%_%v_size%_vectors --name lv-glove-%win%-%v_size%



::python -m spacy init vectors lv ../../Models/SSG_model/ssg_%win%_100_sg.txt ../../Models/Spacy_tagger/ssg_%win%_100_sg_vectors --name lv-ssg-%win%-100