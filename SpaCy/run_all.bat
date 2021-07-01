::This script runs POS and NER training and evaluating scripts and other nescesary scripts for everything in this folder.

call init_all_spacy_vectors.bat 5 50
call train_and_eval_all_ner.bat 5 50
call train_and_eval_all_ner.bat 5 50

call init_all_spacy_vectors.bat 5 100
call train_and_eval_all_ner.bat 5 100
call train_and_eval_all_ner.bat 5 100

call init_all_spacy_vectors.bat 5 200
call train_and_eval_all_ner.bat 5 200
call train_and_eval_all_ner.bat 5 200

call init_all_spacy_vectors.bat 5 300
call train_and_eval_all_ner.bat 5 300
call train_and_eval_all_ner.bat 5 300