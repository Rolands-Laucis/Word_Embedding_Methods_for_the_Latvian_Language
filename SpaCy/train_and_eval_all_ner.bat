::This script trains and evaluates all method word embeddings with the POS task using SpaCy and prepared config-pos.cfg file.

::set up arguments
set win=%1
set v_size=%2

::./train_and_eval_all_ner.bat 5 200

::python -m spacy convert ..\..\datasets\NER\processed\ner-combined-train.iob ./lumii-ner-spacy/ -l lv -n 1
::python -m spacy convert ..\..\datasets\NER\processed\ner-combined-dev.iob ./lumii-ner-spacy/ -l lv -n 1
::python -m spacy convert ..\..\datasets\NER\processed\ner-combined-test.iob ./lumii-ner-spacy/ -l lv -n 1

echo training all NER models
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/ssg_%win%_%v_size%_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/fasttext_%win%_%v_size%_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\fasttext_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/word2vec_%win%_%v_size%_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\word2vec_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/ng2v_%win%_%v_size%_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ng2v_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/glove_%win%_%v_size%-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\glove_%win%_%v_size%_vectors

echo evaluating all NER models
python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_%win%_%v_size%_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/ssg_%win%_%v_size%_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\fasttext_%win%_%v_size%_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/fasttext_%win%_%v_size%_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\word2vec_%win%_%v_size%_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/word2vec_%win%_%v_size%_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\ng2v_%win%_%v_size%_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/ng2v_%win%_%v_size%_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\glove_%win%_%v_size%-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/glove_%win%_%v_size%-ner.json --gold-preproc


::python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/ssg_%win%_100_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_%win%_100_sg_vectors
::python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_%win%_100_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/ssg_%win%_100_sg-ner.json --gold-preproc
