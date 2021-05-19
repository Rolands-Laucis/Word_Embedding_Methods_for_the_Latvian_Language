#this script does NLP tasks for testing created word embeddings using python nlp SpaCy lib. For training POS and NER tagger and using them to evaluate embeddings.

#python nlp_pipe.py

import spacy

#           do NER and morph analisys
nlp = spacy.load("../../Models/Spacy_tagger/fasttext_5_200_sg-ner/model-best/")

#print(nlp.pipeline)
doc = nlp("\"Šeit ir Latvija,\" sacīja deputāts, akcentējot, ka latviešu valoda \"bija, ir un būs vienīgā valsts valoda\".")

for token in doc:
    print(token.text, token.ent_type_, token.ent_iob_)


#           do POS and morph analisys
nlp = spacy.load("../../Models/Spacy_tagger/fasttext_5_200_sg-pos/model-best/")

doc = nlp("Sveiki, šī ir jēdzientelpu apmācības un izmantošanas pamācība.")

print("Word | UPOS | TAG")
for token in doc:
    print(token.text, token.pos_, token.tag_, token.morph)