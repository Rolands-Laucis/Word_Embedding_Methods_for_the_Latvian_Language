#!/bin/sh

#./train_all_ngram2vec.sh
memory_size=4
cpus_num=4
corpus=combined_clean_corpus
output_path=outputs/${corpus}/ngram_ngram

mkdir -p ${output_path}/sgns

python ngram2vec/corpus2vocab.py --corpus_file ${corpus}.txt --vocab_file ${output_path}/vocab --memory_size ${memory_size} --feature ngram 
python ngram2vec/corpus2pairs.py --corpus_file ${corpus}.txt --pairs_file ${output_path}/pairs --vocab_file ${output_path}/vocab --processes_num ${cpus_num} --cooccur ngram_ngram --win 5

# Concatenate pair files. 
if [ -f "${output_path}/pairs" ]; then
	rm ${output_path}/pairs
fi
for i in $(seq 0 $((${cpus_num}-1)))
do
	cat ${output_path}/pairs_${i} >> ${output_path}/pairs
	rm ${output_path}/pairs_${i}
done

# Generate input vocabulary and output vocabulary, which are used as vocabulary files for all models
python ngram2vec/pairs2vocab.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output

# SGNS, learn representation upon pairs.
# We add a python interface upon C code.
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_300_sg.input --output_vector_file ${output_path}/sgns/ng2v_5_300_sg.output --threads_num ${cpus_num} --size 300 --iter 5
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_200_sg.input --output_vector_file ${output_path}/sgns/ng2v_5_200_sg.output --threads_num ${cpus_num} --size 200 --iter 5
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_100_sg.input --output_vector_file ${output_path}/sgns/ng2v_5_100_sg.output --threads_num ${cpus_num} --size 100 --iter 5
#python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_50_sg.input --output_vector_file ${output_path}/sgns/ng2v_5_50_sg.output --threads_num ${cpus_num} --size 50 --iter 5

#everything again for lemmatized
memory_size=4
cpus_num=4
corpus=combined_clean_corpus_lem
output_path=outputs/${corpus}/ngram_ngram

mkdir -p ${output_path}/sgns

python ngram2vec/corpus2vocab.py --corpus_file ${corpus}.txt --vocab_file ${output_path}/vocab --memory_size ${memory_size} --feature ngram 
python ngram2vec/corpus2pairs.py --corpus_file ${corpus}.txt --pairs_file ${output_path}/pairs --vocab_file ${output_path}/vocab --processes_num ${cpus_num} --cooccur ngram_ngram --win 5

# Concatenate pair files. 
if [ -f "${output_path}/pairs" ]; then
	rm ${output_path}/pairs
fi
for i in $(seq 0 $((${cpus_num}-1)))
do
	cat ${output_path}/pairs_${i} >> ${output_path}/pairs
	rm ${output_path}/pairs_${i}
done

# Generate input vocabulary and output vocabulary, which are used as vocabulary files for all models
python ngram2vec/pairs2vocab.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output

# SGNS, learn representation upon pairs.
# We add a python interface upon C code.
#lemmatized:
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_300_sg_lem.input --output_vector_file ${output_path}/sgns/ng2v_5_300_sg_lem.output --threads_num ${cpus_num} --size 300 --iter 5
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_200_sg_lem.input --output_vector_file ${output_path}/sgns/ng2v_5_200_sg_lem.output --threads_num ${cpus_num} --size 200 --iter 5
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_100_sg_lem.input --output_vector_file ${output_path}/sgns/ng2v_5_100_sg_lem.output --threads_num ${cpus_num} --size 100 --iter 5
#python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/ng2v_5_50_sg_lem.input --output_vector_file ${output_path}/sgns/ng2v_5_50_sg_lem.output --threads_num ${cpus_num} --size 50 --iter 5