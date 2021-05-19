#this script cleans the lavian wikipedia data dump .xml file to a clean text file given my regex rules
#https://dumps.wikimedia.org/lvwiki/latest/

#python wikiCleaner.py

from wiki_dump_reader import Cleaner, iterate
import re
from regex_rules import *

i = 0
cleaned_text_string = ""

f = open(r"..\Cleaned_Corpora\cleaned_lvwiki.txt", "w", encoding="utf-8")
f.write("")
f.close()
print("Cleared file")

f = open(r"..\Cleaned_Corpora\cleaned_lvwiki.txt", "a", encoding="utf-8")

print("starting clean")

cleaner = Cleaner()

for title, text in iterate(r"..\Corpora\lvwiki-latest-pages-articles.xml"):
    text = cleaner.clean_text(text)
    cleaned_text, _ = cleaner.build_links(text)
    cleaned_text_string += cleaned_text
    i += 1
    if i%1000 == 0:

        #gets rid of uppercase abreviations
        cleaned_text_string = re.sub('([A-ZĀČĒĢĪĶĻŅŠŪŽ]{2})+', '', cleaned_text_string)

        #lowercase the string
        cleaned_text_string = cleaned_text_string.lower()

        #perform all regex checks on it
        for old, new in RE_replacements_new: #RE_replacements_simple works pretty much the same speed
            cleaned_text_string = re.sub(old, new, cleaned_text_string)

        #write it to file and reset
        f.write(cleaned_text_string)
        cleaned_text_string = ""
        print("Gone through %dk iterations and appended to file" % i)
        #if i >= 10000:
        #    break
    

f.close()

print("cleaned and saved in file!")