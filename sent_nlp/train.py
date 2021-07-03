#this script trains a neural network for text classification on the dataset in ./data into sentiment classes like {positive, negative, neutral...}
#cd sent_nlp
#python train.py

#CODE SNIPPETS CREDIT: https://keras.io/examples/nlp/pretrained_word_embeddings/

#NLP libraries:
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Embedding, Flatten, Activation
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical

#util libs:
import numpy as np
import json
import re

from tensorflow.python.ops.math_ops import argmax

#custom scripts and global variables
from preproc_rules import concatenated_rules
class_names = ["POS", "NEG", "NEU", "IMP", "NOT_LV"]

#--load in tweet text and their sentiments for training data from json:
with open('./data/viksna.json', mode='r', encoding='utf-8') as f:
    data = json.load(f)

    tweets = []
    sentiments = []

    for item in data['data']:
        tweet = item['text']
        #clean the tweet text with my preprosecing
        for rule in concatenated_rules:
            tweet = re.sub(rule[0], rule[1], tweet)
        tweets.append(tweet)

        #get highest value class and append its index to the list of sentiments, e.g. [0,1,1,...]
        sent_vals = [] #= [item['POS'],item['NEG'],item['NEU'],item['IMP'],item['NOT_LV']]
        for c_name in class_names:#support for multiple datasets, so that you can vary the class count
            sent_vals.append(item[c_name])
        sentiments.append(np.argmax(sent_vals))
        #support for all classes having 0-inf counts: (later should not use to_categorical() function on the set)
        #sentiments.append(sent_vals) #e.g. [[0,1,0,0,0,0], [0,0,2,0,0,0],...] <= this gave poor results though
    
    print("First 2 data points from dataset for debug:")
    for i in range(2):
        print("tweet: %s" % tweets[i])
        print("sentiment: %s\n" % sentiments[i])

    print("Gathered %d tweets" % len(tweets))
    print("Gathered %d sentiments" % len(sentiments))
    print("^ These two numbers should be identical.\n\n")

    #clean up memory
    del data


#--set up training and testing datasets for text and labels:
#-split dataset into training and testing sets
training_size_perc = 0.8
training_size = round(len(tweets) * training_size_perc)

training_tweets = np.array(tweets[0:training_size])#to numpy array so the data types match later on
testing_tweets = np.array(tweets[training_size:])

training_sent = to_categorical(np.array(sentiments[0:training_size]), len(class_names)) #to_categorical Converts a class vector (integers) to binary class matrix. I have 3 classes
testing_sent = to_categorical(np.array(sentiments[training_size:]), len(class_names))
#training_sent = np.array(sentiments[0:training_size]) #to_categorical Converts a class vector (integers) to binary class matrix. I have 3 classes
#testing_sent = np.array(sentiments[training_size:])

#-init Tokenizer
tokenizer = Tokenizer(
    num_words=None, #dont perform TF_IDF and dont use min freq for vocabulary, since the dataset is rather small.
    filters='\t\n\r', 
    lower=False, #dont do case folding since it wasnt done for the other tasks either
    split=' ', 
    char_level=False, 
    oov_token='<OOV>' #"out of vocabulary" token. If the model ever encounters an unknown word, it should be labled as this.
)
#clean up memory
del training_size_perc
del training_size


#-build tokenizer vocabulary - assign an int for each found word in training data. This doesnt interfere with custom embeddings; its just to represent strings as number sequences
tokenizer.fit_on_texts(tweets) #here the full tweets is used to minimize OOV words for testing phase of training. This includes words from testing_tweets into the vocabulary.
vocab_size = len(tokenizer.word_index)
print("Tokenizer vocabulary size: %d . First 5 tokenizer vocabulary words:" % vocab_size)
print(list(tokenizer.word_index.items())[:5])

#-create number sequences to train on:
training_tweets_seq = tokenizer.texts_to_sequences(training_tweets)
testing_tweets_seq = tokenizer.texts_to_sequences(testing_tweets)

#save the first 5 tweets for debug at evaluation part of code (peace of mind, that the model works)
first_train_tweets = training_tweets[0:5]
first_test_tweets = testing_tweets[0:5]
#clean up memory
del tweets
del sentiments
del training_tweets
del testing_tweets


#-pad them with 0's so they are all the same length:
training_tweets_seq_pad = pad_sequences(training_tweets_seq, padding='post') #0's are appended to the end reaching the length of the longest sequence in the array
testing_tweets_seq_pad = pad_sequences(testing_tweets_seq, padding='post', maxlen=len(training_tweets_seq_pad[0]))
print("\nFirst 2 padded sequences of the training set for debug:")
print(training_tweets_seq_pad[0:2])
print("All sequences are of length %d" % len(training_tweets_seq_pad[0]))
print("All sequences are of length %d" % len(testing_tweets_seq_pad[0]))
#clean up memory
del training_tweets_seq
del testing_tweets_seq


#-load in embeddings from local file. Need to create them in a matrix form to use with Keras.
print("\nLoading word embeddings...")
embedding_file_path = 'C:/Users/Experimenter/Desktop/BD_Code/Models/ssg_5_100_sg.txt'
embeddings_index = {}
embedding_dimension = 100
errors = 0

with open(embedding_file_path, mode='r', encoding='utf-8') as f:
    for line in f:
        try:
            word, coefs = line.split(' ',maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
        except:
            errors += 1

print("Found %d word vectors. Failed loading %d word vectors." % (len(embeddings_index), errors))
#clean up memory
del errors

#-create the matrix
num_tokens = vocab_size + 1
embedding_matrix = np.zeros((num_tokens, embedding_dimension))
hits = 0
misses = 0

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    input_dim=num_tokens,
    output_dim=embedding_dimension,
    embeddings_initializer= Constant(embedding_matrix),
    trainable=False,
    input_length=len(training_tweets_seq_pad[0])
)

print("Created embedding matrix of shape: %s" % (embedding_matrix.shape,))
#clean up memory
del embedding_file_path
del embeddings_index
del embedding_dimension
del num_tokens
del hits
del misses

#--build model
model = Sequential()
#
model.add(Input(shape=(44,)))
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))#output layer = len(class_names). How many classes there are

model.compile(
    loss='categorical_crossentropy', #sparse_categorical_crossentropy categorical_crossentropy
    optimizer='Adam', 
    metrics=['accuracy']
)

model.summary()

#--train model
print("\nTraining model...")
model.fit(training_tweets_seq_pad, training_sent, epochs=12, verbose=1)

#-evaluate model
print("\nEvaluating model...")

pred_train = model.predict(training_tweets_seq_pad)
scores = model.evaluate(training_tweets_seq_pad, training_sent, verbose=1)
print('Accuracy on training data: %.2f \n Error on training data: %.2f' % (scores[1], 1 - scores[1]))  

# print("Predictions for first few training data tweets")
# for i in range(len(first_train_tweets)):
#     print("\n%s\n%s\n%s" % (first_train_tweets[i],class_names[np.argmax(pred_train[i])], pred_train[i])) 
 
pred_test= model.predict(testing_tweets_seq_pad)
scores2 = model.evaluate(testing_tweets_seq_pad, testing_sent, verbose=1)
print('Accuracy on test data: %.2f \n Error on training data: %.2f' % (scores2[1], 1 - scores2[1])) 

# print("Predictions for first few test data tweets")
# for i in range(len(first_test_tweets)):
#     print("\n%s\n%s\n%s" % (first_test_tweets[i],class_names[np.argmax(pred_test[i])], pred_test[i])) 

#--Save the results to a local file
with open("C:/Users/Experimenter/Desktop/BD_Code/datasets/sentiment/ssg_5_100_sent.txt", mode='w', encoding='utf-8') as output:
    output.write('Accuracy (%%) on train data, then next line - Error on test data\n%.6f\n%.6f' % (scores[1], 1 - scores[1]))
    output.write('\n\nAccuracy (%%) on test data, then next line - Error on test data\n%.6f\n%.6f' % (scores2[1], 1 - scores2[1]))
print("\nSaved results to file.")
