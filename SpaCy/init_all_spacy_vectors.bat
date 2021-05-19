::This script converts all word embeddings to spacy format vectors for use in the training stage of POS and NER models using spacy. Was not used in the thesis, this is just for clarification, since this was done manually.
@echo off

cd ..
::first fasttext and word2vec Gensim .wordvector formats need to be translated to word2vec textual .txt format
python ModelTypeTransform.py --input_file_type wordvectors --input_file ..\Models\Word2vec_model\word2vec_5_200_sg.wordvectors --output_file_type txt --output_file ..\Models\Word2vec_model\word2vec_5_200_sg.txt
python ModelTypeTransform.py --input_file_type wordvectors --input_file ..\Models\FastText_model\fasttext_5_200_sg.wordvectors --output_file_type txt --output_file ..\Models\FastText_model\fasttext_5_200_sg.txt

cd spacy
::then all the word embeddings are converted to spacy format and saved for use in training
python -m spacy init vectors lv ../../Models/FastText_model/fasttext_5_200_sg.txt ../../Models/Spacy_tagger/fasttext_5_200_sg_vectors --name lv-ft-5-200
python -m spacy init vectors lv ../../Models/Word2vec_model/word2vec_5_200_sg.txt ../../Models/Spacy_tagger/word2vec_5_200_sg_vectors --name lv-ssg-5-200
python -m spacy init vectors lv ../../Models/SSG_model/ssg_5_200_sg.txt ../../Models/Spacy_tagger/ssg_5_200_sg_vectors --name lv-ssg-5-200
python -m spacy init vectors lv ../../Models/SSG_model/ssg_5_100_sg.txt ../../Models/Spacy_tagger/ssg_5_100_sg_vectors --name lv-ssg-5-100
python -m spacy init vectors lv ..\..\ngram2vec-master\outputs\combined_clean_corpus\ngram_ngram\sgns\ng2v_5_200_sg.output ../../Models/Spacy_tagger/ng2v_5_200_sg_vectors --name lv-ng2v-5-200

