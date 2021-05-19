#this script is used to lemmatize a plaintext corpus. This is done with UDPipe 1 and the LV model for UDPipe 1. Ofc the python lib is used, but they are bindings to the c++ API

#pip ufal.udpipe

#python lemmatizer.py

import sys
from ufal.udpipe import Model, Pipeline, ProcessingError # pylint: disable=no-name-in-module

i = 0

#load language model - LVTB model
print('Loading model...')
model = Model.load(r'../Models/UDPipe_model/latvian-lvtb-ud-2.5-191206.udpipe')
if not model:
    print('Could not load the model. Exiting.')
    sys.exit(1)
print('Loaded model.')

#Init Pipeline object
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
error = ProcessingError()
#pipeline.set_input('--immediate')
#pipeline.set_immediate(True)

#open and ready taget output file
done_txt = open(r"../Cleaned_Corpora/combined_clean_corpus_lem.txt", "w", encoding="utf-8")
done_txt.write("")
done_txt.close()
print("Cleared file")
done_txt = open(r"../Cleaned_Corpora/combined_clean_corpus_lem.txt", "a", encoding="utf-8")

#iterate over corpus lines and save lemmas in output file
with open('../Cleaned_Corpora/combined_clean_corpus.txt', 'r', encoding='utf-8') as f:
    for line in f:
        processed = pipeline.process(line, error)

        #error handling
        if error.occurred():
            print("An error occured: %s" % error.message)
            sys.exit(1)
        else:
            #for proccessing 'conllu' output format stuff (Kinda looks like a csv file, but i manually parse it):
            lemmatized_line = ''
            for row in processed.split('\n'):
                if (len(row) > 0) and (row[0] != '#'):
                    lemmatized_line += row.split()[2] + ' '
            lemmatized_line += '\n'
            #print(processed)
            #print(processed.split())
            done_txt.write(lemmatized_line)
            i += 1
            if i%10000 == 0:
                print("processed %d lines of text" % i)
        
#close working open file
done_txt.close()
print('done')