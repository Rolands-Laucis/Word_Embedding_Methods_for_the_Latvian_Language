::./trainAllModels.bat 5 100
@echo off

::set up arguments
set win=%1
set v_size=%2

echo training fasttext %win% %v_size%...
python FastTextModeler.py --corpus_file ../Cleaned_Corpora/combined_clean_corpus.txt --window %win% --vector_size %v_size% --method sg --output_file ..\Models\FastText_model\fasttext_%win%_%v_size%_sg.wordvectors
echo training fasttext %win% %v_size% lemmatized...
python FastTextModeler.py --corpus_file ../Cleaned_Corpora/combined_clean_corpus_lem.txt --window %win% --vector_size %v_size% --method sg --output_file ..\Models\FastText_model\fasttext_%win%_%v_size%_sg_lem.wordvectors

echo training word2vec %win% %v_size%...
python word2vecModeler.py --corpus_file ../Cleaned_Corpora/combined_clean_corpus.txt --window %win% --vector_size %v_size% --method sg --output_file ..\Models\Word2vec_model\word2vec_%win%_%v_size%_sg.wordvectors
echo training word2vec %win% %v_size% lemmatized...
python word2vecModeler.py --corpus_file ../Cleaned_Corpora/combined_clean_corpus_lem.txt --window %win% --vector_size %v_size% --method sg --output_file ..\Models\Word2vec_model\word2vec_%win%_%v_size%_sg_lem.wordvectors

::SSG github repo cloned with the folder name wang2vec-master right outside this folder. The tool is C source code, so must be compiled with, e.g. GCC
echo training %win% %v_size% ssg...
cd ..\wang2vec-master 
word2vec -train ../Cleaned_Corpora/combined_clean_corpus.txt -output ../Models/SSG_model/ssg_%win%_%v_size%_sg.txt -type 3 -size %v_size% -window %win% -binary 0
echo training ssg %win% %v_size% lemmatized...
word2vec -train ../Cleaned_Corpora/combined_clean_corpus_lem.txt -output ../Models/SSG_model/ssg_%win%_%v_size%_sg_lem.txt -type 3 -size %v_size% -window %win% -binary 0
cd ..\BD_Word_Embeddings


echo training ngram2vec...
echo ngram2vec should be done seperately
::ngram2vec github repo cloned with the folder name ngram2vec-master right outside this folder. C code must be compiled. 
::Using CygWin i run the ngram_example.sh, which is located in this repo with modifications, but should be placed in /ngram2vec-master
::Also the .sh file expects the corpus .txt's to be in /ngram2vec-master

echo done training models