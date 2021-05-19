::ngram2vecRunner.bat 5 200
@echo off

::set up arguments
set win=%1
set v_size=%2

cd ..\ngram2vec-master\ngram2vec

::python pairs2vocab.py -h

::python corpus2vocab.py --corpus_file ..\..\Cleaned_Corpora\combined_clean_corpus.txt --vocab_file ..\..\Models\Ngram2vec_model\vocab_%win% --feature ngram --order %win%

::python corpus2pairs.py --corpus_file ..\..\Cleaned_Corpora\combined_clean_corpus.txt --pairs_file ..\..\Models\Ngram2vec_model\pairs_%win% --vocab_file ..\..\Models\Ngram2vec_model\vocab_%win% --cooccur ngram_ngram

::change dir to my python script
::cd ..\..\BD_Word_Embeddings

:: concatinate all pairs files into 1 file
::python ngram2vec_pairs_concat.py --window %win% --input_path ../Models/Ngram2vec_model/

::change dir back
::cd ..\ngram2vec-master\ngram2vec

::python pairs2vocab.py --pairs_file ..\..\Models\Ngram2vec_model\pairs_%win%.txt --input_vocab_file ..\..\Models\Ngram2vec_model\vocab_%win%.input --output_vocab_file ..\..\Models\Ngram2vec_model\vocab_%win%.output

python pairs2sgns.py --pairs_file ..\..\Models\Ngram2vec_model\pairs_%win%.txt --input_vocab_file ..\..\Models\Ngram2vec_model\vocab_%win%.input --output_vocab_file ..\..\Models\Ngram2vec_model\vocab_%win%.output --output_vector_file ..\..\Models\Ngram2vec_model\sgns_%win%_%v_size%.output --input_vector_file ..\..\Models\Ngram2vec_model\sgns_%win%_%v_size%.input --size %v_size% --threads_num 2
