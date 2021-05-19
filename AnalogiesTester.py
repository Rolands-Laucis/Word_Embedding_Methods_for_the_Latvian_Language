#This script tests generated word embedding model (word vector space) accuracy on a Latvian analogy data set

#python AnalogiesTester.py --model_type word2vec --model_file ../Models/Word2vec_model/word2vec_5_200_sg_lem.wordvectors --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_word2vec_5_200_3cosmul_top10_lem.txt --gen_output true --eval_method 3cosmul --topn 10
#python AnalogiesTester.py --model_type fasttext_original --model_file ../Models/FastText_model/fasttext_baseline_300.vec --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_fasttext_baseline_300.txt --gen_output true --eval_method %eval_method% --topn 1
#python AnalogiesTester.py --model_type ssg --model_file ../Models/SSG_model/ssg_model.txt --dataset_file ../datasets/lv-analogies.txt --output_file ../datasets/results_ssg.txt --gen_output true

import argparse
from gensim.models import Word2Vec, KeyedVectors
import time

def main():
    start = time.time()
    #CLI argument handling
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_type", type=str, required=True, help="[word2vec|fasttext_original|fasttext_gensim|ngram2vec|ssg]")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the .wordvectors or .txt file.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset txt file.")
    parser.add_argument("--eval_method", type=str, required=True, help="[gensim|3cosmul|3cosadd]")
    parser.add_argument("--topn", type=int, default=1, help="Answer accepted, if in topn results from similarity check. Default 1.")
    parser.add_argument("--dummy4unknown", type=bool, default=True, help="Should the analogy line be skipped, if not all words in vocabulary? Default True.")
    parser.add_argument("--verbose", type=bool, default=False, help="Should the program give status updates? Default False.")
    parser.add_argument("--gen_output", type=bool, default=True, help="Should the output file with score and incorrect guesses be generated. Default True.")
    parser.add_argument("--output_file", type=str, default=r'../datasets/results.txt', help="Path to the output txt file.")
    args = parser.parse_args()

    if args.gen_output:
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
    elif args.model_type == "fasttext_gensim":
        #loads gensim FastText generated .wordvectors file
        word_vectors = KeyedVectors.load(args.model_file)
    elif args.model_type == "ssg" or args.model_type == "ngram2vec":
        word_vectors = KeyedVectors.load_word2vec_format(args.model_file, binary=False)
    else:
        print("model_type not supported!")
        return
    print("model loaded!")

    #evaluate the model by different methods
    score = 0
    sections = None

    if args.eval_method == "gensim":
        score, sections = word_vectors.evaluate_word_analogies(args.dataset_file, case_insensitive=True, dummy4unknown=args.dummy4unknown)
    elif args.eval_method == "3cosmul" or args.eval_method == "3cosadd":
        score, sections = AnalogyEval(args.dataset_file, word_vectors, args.eval_method, args.topn, args.dummy4unknown, args.verbose, args.model_type)
    else:
        print("eval_method not supported!")
        return

    
    #output evaluation results
    if args.gen_output:
        output = open(args.output_file, "a", encoding='utf-8')
        output.write("%1.6f,%s,%s,%d,%r\n" % (score*100, args.model_type, args.eval_method, args.topn, args.dummy4unknown))
        if args.eval_method == "3cosmul" or args.eval_method == "3cosadd":
            for key, value in sections.items():
                #print("%s,%1.6f\n" % (key, value))
                output.write("%s,%1.6f\n" % (key, value))
        #output.write("Dataset evaluated with score: %1.6f%% and parameters %s %s %d %r \n incorrect guesses: \n" % (score*100, args.model_type, args.eval_method, args.topn, args.dummy4unknown))

        if False:
            if args.eval_method == "gensim":
                for section in sections:
                    for line in section['incorrect']:
                        for word in line:
                            output.write("%s " % word)
                        output.write('\n')
            elif args.eval_method == "3cosmul" or args.eval_method == "3cosadd":
                for line in sections['incorrect']:
                        for word in line:
                            output.write("%s " % word)
                        #print(line)
                        output.write('\n')
        output.close()
    
    print("Dataset evaluated with score: %1.6f in %.2f min" % (score, ((time.time()-start)/60)))
    #print("Dataset evaluated!")


#evaluates a analogy dataset .txt file and returns a score (0-1) of how many were answered correctly and scores per section
def AnalogyEval(file, word_vectors, method, top, dummy4unknown, verbose, model_type):
    analogies_proccessed = 0
    correct_answers = 0
    correct_in_category = 0
    analogies_in_category = 0

    section_name = ""
    sections = {}

    #go through analogies file line by line and check the 4 word analogy evaluation
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            #get all 4 words from line
            if model_type == "fasttext_original":# because fasttext baseline doesnt do case folding for its embeddings, i dont case fold analogy words either
                words = line.split()
            else:
                words = [x.lower() for x in line.split()] #btw no need to lower them, if using a prebaked analogy file for this method. But usually i dont use the prebaked

            #if this line describes a new category, not 4 words of analogies
            if (":" in words):
                if section_name == "":
                    section_name = words[1]
                    #section_name = " ".join(words)
                else:
                    if analogies_in_category != 0:
                        sections[section_name] = (correct_in_category/analogies_in_category) * 100
                    else:
                        sections[section_name] = 0
                    correct_in_category = 0
                    analogies_in_category = 0
                    section_name = words[1]
                    #section_name = "_".join(words)
                continue

            #check if all required words are in vocabulary - dummy4unknown
            if dummy4unknown:
                skip_line = False
                for n in range(4):
                    if word_vectors.has_index_for(words[n]) == False:
                        skip_line = True
                        break
                if skip_line:
                    continue
            
            #get answer or answers
            answer = None

            #choose evaluation method that was passed in
            if method == "3cosmul":
                answer = word_vectors.most_similar_cosmul(positive=[words[1],words[2]], negative=[words[0]], topn=top)
            elif method == "3cosadd":
                answer = word_vectors.most_similar(positive=[words[1],words[2]], negative=[words[0]], topn=top)
            
            #count score
            if words[3] in dict(answer):
                correct_answers += 1
                correct_in_category += 1
            analogies_proccessed += 1
            analogies_in_category += 1

            #status update and/or break condition
            if verbose:
                if analogies_proccessed%100 == 0:
                    print("processed %d lines of analogies" % analogies_proccessed)
                    #print(words)
                    #print(dict(answer))
                    #break
    if analogies_in_category != 0:
        sections[section_name] = (correct_in_category/analogies_in_category) * 100
    else:
        sections[section_name] = 0
    return (correct_answers/analogies_proccessed), sections

if __name__ == '__main__':
    main()