#this script gathers information about a corpus like total word count, vocab size, vocab size with a min freq

#with GloVe vocab file:
#python corpus_info.py --file_path ..\GloVe-master\vocab.txt --output_file ..\datasets\corpus_info.txt --min_freq 10
#python corpus_info.py --file_path ..\GloVe-master\vocab_lem.txt --output_file ..\datasets\corpus_lem_info.txt --min_freq 10

#python corpus_info.py --count_method 1 --file_path ..\Cleaned_Corpora\combined_clean_corpus.txt --output_file ..\datasets\corpus_info_custom.txt --min_freq 10
#python corpus_info.py --count_method 1 --file_path ..\Cleaned_Corpora\combined_clean_corpus_lem.txt --output_file ..\datasets\corpus_info_lem_custom.txt --min_freq 10

import argparse

def main():
    #CLI argument handling
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("--vocab_file", type=str, required=False, help="Path to ngram2vec vocab file, since it already has computed all the info i need.")
    parser.add_argument("--count_method", type=int, default=1, help="0=vocab file; 1=corpus file; Default 1")
    parser.add_argument("--file_path", type=str, required=True, help="Path to GloVe vocab file or corpus file path, if using custom counting method.")
    parser.add_argument("--min_freq", type=int, default=10, help="Minimum word frequency in corpus to be counted here in the total vocab size. Default 10")
    parser.add_argument("--output_file", type=str, default=r'../datasets/corpus_info.txt', help="Path to the output txt file.")
    args = parser.parse_args()

    #clear output file for new output data
    output = open(args.output_file, "w", encoding='utf-8')
    output.write("%s <= min: %d\n" % (args.file_path, args.min_freq))
    output.close()
    print("Cleared dataset output file")
    output = open(args.output_file, "a", encoding='utf-8')

    total_word_count = 0
    vocab_size = 0
    vocab_size_of_min = 0

    if args.count_method == 0:
        with open(args.file_path, mode='r', encoding='utf8') as f:
            for line in f:
                line = line.split(' ')
                word_count = int(line[1])
                
                total_word_count += word_count
                vocab_size += 1
                if word_count >= args.min_freq:
                    vocab_size_of_min += 1
            
    else:
        vocab = {}
        with open(args.file_path, mode='r', encoding='utf8') as f:
            for line in f:
                words = line.split(' ')
                total_word_count += len(words)
                for word in words:
                    if vocab.get(word) != None:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        vocab_size = len(vocab)
        for (key, value) in vocab.items():
            if value >= args.min_freq:
                vocab_size_of_min += 1
    
    output.write("Total words in corpus | vocab size | vocab size with min freq\n")
    output.write("%d %d %d" % (total_word_count, vocab_size, vocab_size_of_min))
    output.close()
    print("Done generating corpus info!")


if __name__ == '__main__':
    main()