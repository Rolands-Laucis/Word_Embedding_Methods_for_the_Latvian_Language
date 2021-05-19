::./analogyEval.bat 5 200 3cosmul
@echo off

::set up arguments
set win=%1
set v_size=%2
set eval_method=%3

::pause
::exit 0

echo doing word2vec %win% %v_size% analogy evaluations
python AnalogiesTester.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_%win%_%v_size%_sg.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_word2vec_%win%_%v_size%_%eval_method%_top10.txt --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_%win%_%v_size%_sg.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_word2vec_%win%_%v_size%_%eval_method%.txt --eval_method %eval_method% --topn 1
echo doing word2vec %win% %v_size% analogy evaluations on lemmatized corpora
python AnalogiesTester.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_%win%_%v_size%_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_word2vec_%win%_%v_size%_%eval_method%_top10_lem.txt --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_%win%_%v_size%_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_word2vec_%win%_%v_size%_%eval_method%_lem.txt --eval_method %eval_method% --topn 1

echo doing FastText %win% %v_size% analogy evaluations
python AnalogiesTester.py --model_type fasttext_gensim --model_file ../Models/FastText_model/fasttext_%win%_%v_size%_sg.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_%win%_%v_size%_%eval_method%_top10.txt --gen_output true --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type fasttext_gensim --model_file ../Models/FastText_model/fasttext_%win%_%v_size%_sg.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_%win%_%v_size%_%eval_method%.txt --gen_output true --eval_method %eval_method% --topn 1
echo doing FastText %win% %v_size% analogy evaluations on lemmatized corpora
python AnalogiesTester.py --model_type fasttext_gensim --model_file ../Models/FastText_model/fasttext_%win%_%v_size%_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_%win%_%v_size%_%eval_method%_top10_lem.txt --gen_output true --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type fasttext_gensim --model_file ../Models/FastText_model/fasttext_%win%_%v_size%_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_%win%_%v_size%_%eval_method%_lem.txt --gen_output true --eval_method %eval_method% --topn 1

::fasttext baseline original vectors for latvian
python AnalogiesTester.py --model_type fasttext_original --model_file ../Models/FastText_model/fasttext_baseline_300.vec --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_baseline_300.txt --gen_output true --eval_method %eval_method% --topn 1
python AnalogiesTester.py --model_type fasttext_original --model_file ../Models/FastText_model/fasttext_baseline_300.vec --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_baseline_300_top10.txt --gen_output true --eval_method %eval_method% --topn 10


echo doing SSG %win% %v_size% analogy evaluations
python AnalogiesTester.py --model_type ssg --model_file ../Models/SSG_model/ssg_%win%_%v_size%_sg.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ssg_%win%_%v_size%_%eval_method%_top10.txt --gen_output true --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type ssg --model_file ../Models/SSG_model/ssg_%win%_%v_size%_sg.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ssg_%win%_%v_size%_%eval_method%.txt --gen_output true --eval_method %eval_method% --topn 1
echo doing SSG %win% %v_size% analogy evaluations on lemmatized corpora
python AnalogiesTester.py --model_type ssg --model_file ../Models/SSG_model/ssg_%win%_%v_size%_sg_lem.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ssg_%win%_%v_size%_%eval_method%_top10_lem.txt --gen_output true --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type ssg --model_file ../Models/SSG_model/ssg_%win%_%v_size%_sg_lem.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ssg_%win%_%v_size%_%eval_method%_lem.txt --gen_output true --eval_method %eval_method% --topn 1


echo doing ngram2vec analogy evaluations
::there is an option for word-word coocurance in ngram2vec, but from the paper it seems like its just SGNS. So only ngram-ngram with default settings and window = 5 models are used and tested (nn - ngram-ngram)
python AnalogiesTester.py --model_type ngram2vec --model_file ../ngram2vec-master/outputs/combined_clean_corpus/ngram_ngram/sgns/ng2v_%win%_%v_size%_sg.output --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ng2v_%win%_%v_size%_%eval_method%_nn_top10.txt --gen_output true --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type ngram2vec --model_file ../ngram2vec-master/outputs/combined_clean_corpus/ngram_ngram/sgns/ng2v_%win%_%v_size%_sg.output --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ng2v_%win%_%v_size%_%eval_method%_nn.txt --gen_output true --eval_method %eval_method% --topn 1
echo doing ngram2vec analogy evaluations on lemmatized corpora
python AnalogiesTester.py --model_type ngram2vec --model_file ../ngram2vec-master/outputs/combined_clean_corpus_lem/ngram_ngram/sgns/ng2v_%win%_%v_size%_sg_lem.output --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ng2v_%win%_%v_size%_%eval_method%_nn_top10_lem.txt --gen_output true --eval_method %eval_method% --topn 10
python AnalogiesTester.py --model_type ngram2vec --model_file ../ngram2vec-master/outputs/combined_clean_corpus_lem/ngram_ngram/sgns/ng2v_%win%_%v_size%_sg_lem.output --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ng2v_%win%_%v_size%_%eval_method%_nn_lem.txt --gen_output true --eval_method %eval_method% --topn 1

echo %win% %v_size% %eval_method% analogy evaluations done