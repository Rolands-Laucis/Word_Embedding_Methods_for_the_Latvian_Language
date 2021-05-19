@echo off

echo training all POS models
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/ssg_5_200_sg-pos --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_5_200_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/ssg_5_100_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_5_100_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/fasttext_5_200_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\fasttext_5_200_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/word2vec_5_200_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\word2vec_5_200_sg_vectors
python -m spacy train ./configs/config-pos.cfg --output ../../Models/Spacy_tagger/ng2v_5_200_sg-pos --paths.train --paths.train ./lvtb-pos-spacy/lv_lvtb-ud-train.spacy --paths.dev ./lvtb-pos-spacy/lv_lvtb-ud-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ng2v_5_200_sg_vectors

echo evaluating all POS models
python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_5_200_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/ssg_5_200_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_5_100_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/ssg_5_100_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\fasttext_5_200_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/fasttext_5_200_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\word2vec_5_200_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/word2vec_5_200_sg-pos.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\ng2v_5_200_sg-pos\model-best ./lvtb-pos-spacy/lv_lvtb-ud-test.spacy --output ../../datasets/POS/ng2v_5_200_sg-pos.json --gold-preproc