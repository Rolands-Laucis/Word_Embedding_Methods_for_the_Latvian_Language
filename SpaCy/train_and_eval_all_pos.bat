::This script trains and evaluates all method word embeddings with the POS task using SpaCy and prepared config-pos.cfg file.

::set up arguments
set win=%1
set v_size=%2

echo training all POS models
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/ssg_%win%_%v_size%_sg-pos --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/fasttext_%win%_%v_size%_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\fasttext_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/word2vec_%win%_%v_size%_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\word2vec_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/ng2v_%win%_%v_size%_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ng2v_%win%_%v_size%_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/glove_%win%_%v_size%-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\glove_%win%_%v_size%_vectors


echo evaluating all POS models
python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_%win%_%v_size%_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/ssg_%win%_%v_size%_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\fasttext_%win%_%v_size%_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/fasttext_%win%_%v_size%_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\word2vec_%win%_%v_size%_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/word2vec_%win%_%v_size%_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\ng2v_%win%_%v_size%_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/ng2v_%win%_%v_size%_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\glove_%win%_%v_size%-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/glove_%win%_%v_size%-pos.json --gold-preproc


::python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/ssg_%win%_100_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_%win%_100_sg_vectors
::python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_%win%_100_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/ssg_%win%_100_sg-pos.json --gold-preproc
