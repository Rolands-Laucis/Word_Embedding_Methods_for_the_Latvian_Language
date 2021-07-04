::train and eval all embeddings in the sentiment analysis NLP task
::./train_all.bat 5 50

@echo off

::set up CLI arguments
set win=%1
set v_size=%2

::cd ..
::python ModelTypeTransform.py --input_file_type wordvectors --input_file ..\Models\fasttext_5_%v_size%_sg.wordvectors --output_file_type txt --output_file ..\Models\fasttext_5_%v_size%_sg.txt
::cd sent_nlp
::exit /b 0
python train.py --embeddings_path ..\..\Models\fasttext_5_%v_size%_sg.txt --embeddings_dim %v_size% --output_file ..\..\datasets\sentiment\fasttext_5_%v_size%_sent.txt
exit /b 0

echo training SSG sentiment analysis model...
python train.py --embeddings_path ..\..\Models\ssg_5_%v_size%_sg.txt --embeddings_dim %v_size% --output_file ..\..\datasets\sentiment\ssg_5_%v_size%_sent.txt
echo training GloVe sentiment analysis model...
python train.py --embeddings_path ..\..\Models\glove_5_%v_size%.txt --embeddings_dim %v_size% --output_file ..\..\datasets\sentiment\glove_5_%v_size%_sent.txt
exit /b 0

python train.py --embeddings_path ..\..\Models\fasttext_5_%v_size%_sg.txt --embeddings_dim %v_size% --output_file ..\..\datasets\sentiment\fasttext_5_%v_size%_sent.txt
python train.py --embeddings_path ..\..\Models\ssg_5_%v_size%_sg.txt --embeddings_dim %v_size% --output_file ..\..\datasets\sentiment\ssg_5_%v_size%_sent.txt
python train.py --embeddings_path ..\..\Models\ssg_5_%v_size%_sg.txt --embeddings_dim %v_size% --output_file ..\..\datasets\sentiment\ssg_5_%v_size%_sent.txt