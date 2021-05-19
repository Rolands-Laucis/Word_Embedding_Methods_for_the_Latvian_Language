@echo off

echo preproccessing and creating text corpora
python wikiCleaner.py
python europarlCorpus.py
python combineCorpora.py
python lemmatizer.py
pause