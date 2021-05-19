#this script prepares an analogy.txt file per embedding method that only contains analogies, where every word is in that methods models vocabullary.
#basically this speeds up the evaluations because the dummy4unknown check is prebaked with this. EDIT: actually the speed impact from doing this is ineffectively small.
#But this script is also used to just get different counts of the analogies file.

#python analogyPrep.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_5_200_sg.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-word2vec.txt
#python analogyPrep.py --model_type fasttext --model_file ../Models/FastText_model/fasttext_5_200_sg.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-fasttext.txt
#python analogyPrep.py --model_type ssg --model_file ../Models/SSG_model/ssg_5_200_sg.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-ssg.txt
#python analogyPrep.py --model_type ngram2vec --model_file ../ngram2vec-master/outputs/combined_clean_corpus/ngram_ngram/sgns/ng2v_5_200_sg.output --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-ngram2vec.txt

#python analogyPrep.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_5_200_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-word2vec_lem.txt
#python analogyPrep.py --model_type fasttext --model_file ../Models/FastText_model/fasttext_5_200_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-fasttext_lem.txt
#python analogyPrep.py --model_type ssg --model_file ../Models/SSG_model/ssg_5_200_sg_lem.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-ssg_lem.txt
#python analogyPrep.py --model_type ngram2vec --model_file ../ngram2vec-master/outputs/combined_clean_corpus_lem/ngram_ngram/sgns/ng2v_5_200_sg_lem.output --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-ngram2vec_lem.txt

#python analogyPrep.py --model_type fasttext_original --model_file ../Models/FastText_model/fasttext_baseline_300.vec --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/lv-analogies-fasttext-baseline.txt

import argparse
from gensim.models import Word2Vec, KeyedVectors

def main():
    #CLI argument handling
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_type", type=str, required=True, help="[word2vec|fasttext|ngram2vec|ssg]")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the .wordvectors or .txt file.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset txt file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output txt file.")
    args = parser.parse_args()

    #clear output file for new output data
    output = open(args.output_file, "w", encoding='utf-8')
    output.write("")
    output.close()
    print("Cleared dataset output file")
    output = open(args.output_file, "a", encoding='utf-8')

    #load model as a Wordvectors object
    if args.model_type == "word2vec":
        word_vectors = KeyedVectors.load(args.model_file)
    elif args.model_type == "fasttext_original":
        #loads fasttext.cc pretrained .vec file
        word_vectors = KeyedVectors.load_word2vec_format(args.model_file, binary=False)
    elif args.model_type == "fasttext":
        #loads gensim FastText generated .wordvectors file
        word_vectors = KeyedVectors.load(args.model_file)
    elif args.model_type == "ssg" or args.model_type == "ngram2vec":
        word_vectors = KeyedVectors.load_word2vec_format(args.model_file, binary=False)
    else:
        print("model_type not supported!")
        return
    print("model loaded!")


    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if args.model_type == "fasttext_original":# because fasttext baseline doesnt do case folding for its embeddings, i dont case fold analogy words either
                words = line.split()
            else:
                words = [x.lower() for x in line.split()] #get all 4 words from line
            if (":" in words):
                #output.write(" ".join(words) + "\n")
                continue

            skip_line = False
            for n in range(4):
                if word_vectors.has_index_for(words[n]) == False:
                    skip_line = True
                    break
            if skip_line:
                continue
            else:
                output.write(" ".join(words) + "\n")
    
    print("analogy file for %s created!" % args.model_type)

            

if __name__ == '__main__':
    main()