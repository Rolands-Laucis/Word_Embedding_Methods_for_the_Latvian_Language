#this script implements the fastText python library to create a word embeddings language model from my corpus using FastText method with gensim.

#python FastTextModeler.py --corpus_file ../Cleaned_Corpora/combined_clean_corpus.txt --window 5 --vector_size 200 --method sg

import argparse
from gensim.models import FastText
import time

def main():
    start = time.time()
    #CLI arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus_file", type=str, required=True, help="Path to the corpus .txt file.")
    parser.add_argument("--window", type=int, default=5, help="Window size. default=5")
    parser.add_argument("--vector_size", type=int, default=200, help="Vector dimensions. default=200")
    parser.add_argument("--min_count", type=int, default=1, help="The model ignores all words with total frequency lower than this. default=1")
    parser.add_argument("--method", type=str, required=True, help="[sg|cbow]")
    parser.add_argument("--output_file", type=str, default=r'../Models/FastText_model/FastText_combined_corp_model.wordvectors', help="Path to the output .wordvectors file.")
    args = parser.parse_args()

    #set method
    method = 1
    if args.method == "sg":
        method = 1
    elif args.method == "cbow":
        method = 0
    else:
        print("method not supported!")
        return

    #setup model
    model = FastText(vector_size=args.vector_size, window=args.window, min_count=args.min_count, sg=method)
    model.build_vocab(corpus_file=args.corpus_file)
    print("Built vocab. Starting train...")

    #train
    model.train(corpus_file=args.corpus_file, total_words=model.corpus_total_words, epochs=5)
    model.wv.save(args.output_file)
    print ("Done training fasttext in %.2f min" % ((time.time()-start)/60))

if __name__ == '__main__':
    main()