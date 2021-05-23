#this script parses LUMII-AILab NER dataset files (https://github.com/LUMII-AILab/FullStack/tree/master/NamedEntities) into 1 .iob file (currently TildeNER is excluded from the end file, because of SpaCy issues) and then into 3 .iob files (train, dev, test), split by sentence % 60-30-10
#file format generated looks exactly like spaCy -c ner argument takes in this example file: 
#https://github.com/explosion/spaCy/blob/df3444421aba611d4ad1238610ce189df158d85a/extra/example_data/ner_example_data/ner-token-per-line-conll2003.iob

#python ner_file_concat.py

import glob
import math

output_file_paths_conll2003 = ["..\..\datasets\\NER\data\\"]
output_conll2003_path = "..\..\datasets\\NER\processed\\LUMII.iob"

#TildeNER is currently excluded at the last statge of creating the big .iob file, because of SpaCy converter issues. So this may be ignored. This script has not been tested being run with these extra lines being commented out. TildeNER was excluded (commented out) at the final concatination stage of creating the big .iob file.
#input_file_paths_gold = ["..\..\datasets\\NER\TEST\dev_in\\", "..\..\datasets\\NER\TEST\seed_in\\", "..\..\datasets\\NER\TEST\gold_tab_sep_in\\"]
#output_gold_path = "..\..\datasets\\NER\processed\\TildeNER.iob"

combined_iob_file_path = "..\..\datasets\\NER\processed\\ner-combined.iob"
output_file_paths = ['..\..\datasets\\NER\\processed\\ner-combined-train.iob', '..\..\datasets\\NER\\processed\\ner-combined-dev.iob', '..\..\datasets\\NER\\processed\\ner-combined-test.iob']

#pos_translation = {'?':'X','s':'X','p':'X','V':'VERB','Q':'NUM','P':'ADP','O':'PRON','N':'NOUN','I':'INTJ','D':'X','C':'CCONJ','B':'ADV','A':'ADJ', '-':'X'}
#pos_translation = {'SENT':'X','-':'X','Z':'PUNCT','X':'X','W':'X','V':'VERB','I':'INTJ','S':'SCONJ','T':'PART','D':'ADV','P':'PROPN','N':'NOUN','G':'PART','A':'ADJ','R':'ADP','F':'X','B':'DET','C':'NUM','H':'CCONJ',}

def NewOutputFile(path, final=False):
    #clear output file for new output data
    output_f = open(path, "w", encoding='utf-8')
    if final:
        output_f.write("-DOCSTART- -X- O O\n")
    else:
        output_f.write("")
    output_f.close()
    print("Cleared %s dataset output file" % path)
    output_f = open(path, "a", encoding='utf-8')
    return output_f

def ProcessFiles(paths, f_type):
    files_processed = 0

    for path in paths:
        for found_file in glob.glob(path + "*."+f_type):
            with open(found_file, mode='r', encoding='utf8') as f:
                for line in f:
                    tokens = line.split('\t')
                    if line[0] == '#':
                        continue
                    elif line[0] == '\n':
                        output.write(line)
                    else:
                        out_line = ''
                        if f_type == 'gold':
                            #so someone else before me has made a mistake in creating these files. There is supposed to be 1 word token per line as per documentation. 
                            #I found places where there are 2 words in a token per line. 
                            #This took me 2.5 hours of frustration.
                            #.gold and .conll2003 NER and .IOB are not my favorite file formats.
                            #
                            #I ignore lines, where the first token (word) is actually 2 words seperated by a space.
                            if ' ' in tokens[0]:
                                continue
                            out_line = " ".join([tokens[0], tokens[8]]) #POS is tokens[1] pos_translation[tokens[1]]
                        elif f_type == 'conll2003':
                            if ' ' in tokens[1]:
                                continue
                            out_line = " ".join([tokens[1], tokens[6]]) + "\n" #POS is tokens[3]; tokens[6] is BIOTAG1 which is the outter entity tag.
                        #print(out_line)
                        output.write(out_line)

            files_processed += 1
            if files_processed%1000 == 0:
                print("processed %d files..." % files_processed)


#                       ---MAIN---

#combine all .gold files into 1 long .iob file with only the token and IOB fields
#output = NewOutputFile(output_gold_path)
#ProcessFiles(input_file_paths_gold, 'gold')
#output.close()

#combine all .conll2003 files into 1 long .iob file with only the token and IOB fields
output = NewOutputFile(output_conll2003_path)
ProcessFiles(output_file_paths_conll2003, 'conll2003')
output.close()

#combine the 2 .iob files into 1 long .iob file
output = NewOutputFile(combined_iob_file_path)
with open(output_conll2003_path, mode="r", encoding='utf8') as f:
    for line in f:
        output.write(line)

#excluding the TildeNER set because i have a lot of problems with its format. Very confusing. 
#And from all the tests it looks like models get ~5% F-score on NER when it is included. Bad files. Lots of mistakes in them as well.
#with open(output_gold_path, mode="r", encoding='utf8') as f:
#    for line in f:
#        output.write(line)
output.close()

#split the .iob long file into train, dev, test files

#count how many sentences should be in each file for a % split of 60-30-10 as seen in the recomended quide lines online
sentence_count = 0

with open(combined_iob_file_path, mode="r", encoding='utf8') as f:
    for line in f:
        if line[0] == '\n':
            sentence_count += 1

train_sent_count = math.floor(sentence_count * 0.6)
dev_sent_count = math.floor(sentence_count * 0.3)
test_sent_count = math.floor(sentence_count * 0.1)

#loop through all 3 output files and copy paste the lines in them
sentence_count = 0

with open(combined_iob_file_path, mode="r", encoding='utf8') as f:

    output = NewOutputFile(output_file_paths[0], True)
    for line in f:
        #at each new sentence check if need to swap to a different file
        if line[0] == '\n':
            sentence_count += 1
            if sentence_count == train_sent_count:
                output.close()
                output = NewOutputFile(output_file_paths[1], True)
                check_sent_count = False
            elif sentence_count == train_sent_count + dev_sent_count:
                output.close()
                output = NewOutputFile(output_file_paths[2], True)
                check_sent_count = False

        #just copy paste the line to the file
        output.write(line)
    output.close()

print("Done! Total sentences in final file: %d, in train: %d, in dev: %d, in test: %d" % (sentence_count, train_sent_count, dev_sent_count, test_sent_count))

#if __name__ == '__main__':
#    main()
