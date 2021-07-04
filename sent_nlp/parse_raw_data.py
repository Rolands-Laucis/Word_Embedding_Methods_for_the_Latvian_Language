import numpy as np
import pandas as pd
import json
import re
from preproc_rules import concatenated_rules

def Parse(file_path, class_names, verbose):
    with open(file_path, mode='r', encoding='utf8') as f:
        data = json.load(f) #into python dict
        #data = pd.read_csv(file_path,sep=',',usecols=['text', 'sumsent'], encoding='utf8')

        tweets = []
        sentiments = []
        errors = 0

        #for item in data['data']:
        for item in data[0]:
        #for index, item in data.iterrows():
            #print(item)
            try:
                tweet = item['text']
                #clean the tweet text with my preprosecing
                for rule in concatenated_rules:
                    tweet = re.sub(rule[0], rule[1], tweet)
                tweets.append(tweet)

                #get highest value class and append its index to the list of sentiments, e.g. [0,1,1,...]
                #sent_vals = [] #= [item['POS'],item['NEG'],item['NEU'],item['IMP'],item['NOT_LV']]
                #for c_name in class_names:#support for multiple datasets, so that you can vary the class count
                #    sent_vals.append(item[c_name])
                #sentiments.append(np.argmax(sent_vals))

                #support for all classes having 0-inf counts: (later should not use to_categorical() function on the set)
                #sentiments.append(sent_vals) #e.g. [[0,1,0,0,0,0], [0,0,2,0,0,0],...] <= this gave poor results though

                if item['sumsent'] >= 1:
                    sentiments.append(0)
                elif item['sumsent'] <= -1:
                    sentiments.append(2)
                else:
                    sentiments.append(1)
            except:
                errors += 1
        if verbose:
            #print(data.head(2))
            print("First 2 data points from dataset for debug:")
            for i in range(2):
                print("tweet: %s" % tweets[i])
                print("sentiment: %s\n" % sentiments[i])

        print("Gathered %d tweets" % len(tweets))
        print("Gathered %d sentiments" % len(sentiments))
        print("^ These two numbers should be identical.")
        print("Errors: %d" % errors)

        #clean up memory
        del data

        return tweets, sentiments