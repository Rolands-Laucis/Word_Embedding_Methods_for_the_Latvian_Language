#using the euro parl http://www.statmt.org/europarl/ lv-en corpus, but only the lv file
#this file cleans up the original downloaded corpus file given my regex rules

#python europarlCorpus.py

from regex_rules import *
import re

i = 0
cleaned_text_string = ""

done_txt = open(r"..\Cleaned_Corpora\cleaned_europarl_lv.txt", "w")
done_txt.write("")
done_txt.close()
print("Cleared file")

done_txt = open(r"..\Cleaned_Corpora\cleaned_europarl_lv.txt", "a", encoding="utf-8")

print("starting clean")

with open(r"..\Corpora\europarl-v7-lv.txt", "r", encoding="utf-8") as f:
    for line in f:
        i += 1

        #for europarl:
        line = re.sub('([A-ZĀČĒĢĪĶĻŅŠŪŽ]{2})+', '', line)
        #gets rid of uppercase abreviations

        line = line.lower()
        for old, new in RE_replacements_new:
            line = re.sub(old, new, line)

        done_txt.write(line)
        if i%10000 == 0:
            print("processed %d lines of text" % i)

done_txt.close()
print('done')