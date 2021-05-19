@echo off

::./train_and_eval_all_ner.bat

::python -m spacy convert ..\..\datasets\NER\processed\ner-combined-train.iob ./lumii-ner-spacy/ -l lv -n 1
::python -m spacy convert ..\..\datasets\NER\processed\ner-combined-dev.iob ./lumii-ner-spacy/ -l lv -n 1
::python -m spacy convert ..\..\datasets\NER\processed\ner-combined-test.iob ./lumii-ner-spacy/ -l lv -n 1

echo training all NER models
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/ssg_5_200_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_5_200_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/ssg_5_100_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ssg_5_100_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/fasttext_5_200_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\fasttext_5_200_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/word2vec_5_200_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\word2vec_5_200_sg_vectors
python -m spacy train ./configs/config-ner.cfg --output ../../Models/Spacy_tagger/ng2v_5_200_sg-ner --paths.train ./lumii-ner-spacy/ner-combined-train.spacy --paths.dev ./lumii-ner-spacy/ner-combined-dev.spacy --paths.vectors ..\..\Models\Spacy_tagger\ng2v_5_200_sg_vectors

echo evaluating all NER models
python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_5_200_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/ssg_5_200_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\ssg_5_100_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/ssg_5_100_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\fasttext_5_200_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/fasttext_5_200_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\word2vec_5_200_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/word2vec_5_200_sg-ner.json --gold-preproc
python -m spacy evaluate ..\..\Models\Spacy_tagger\ng2v_5_200_sg-ner\model-best ./lumii-ner-spacy/ner-combined-test.spacy --output ../../datasets/NER/ng2v_5_200_sg-ner.json --gold-preproc