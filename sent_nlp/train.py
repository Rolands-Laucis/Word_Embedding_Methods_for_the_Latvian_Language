#this script trains a neural network for text classification on the dataset in ./data into sentiment classes {positive, negative, neutral}
#cd sent_nlp
#python train.py

#CODE CREDIT: https://orbifold.net/default/embedding-and-tokenizer-in-keras/
#https://www.youtube.com/watch?v=Y_hzMnRXjhI

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

#custom scripts
from preproc_rules import concatenated_rules

#--load in tweet text and their sentiments for training data from json:
with open('./data/viksna.json', mode='r', encoding='utf-8') as f:
    data = json.load(f)

    tweets = []
    sentiments = []

    for item in data['data']:
        tweet = item['text']
        #clean the text with my preprosecing
        for rule in concatenated_rules:
            tweet = re.sub(rule[0], rule[1], tweet)

        tweets.append(tweet)
        if item['POS'] == 1:
            sentiments.append(2)
        elif item['NEG'] == 1:
            sentiments.append(0)
        else:
            sentiments.append(1)
    
    print("First 2 data points from dataset for debug:")
    for i in range(2):
        print("tweet: %s" % tweets[i])
        print("sentiment: %d\n" % sentiments[i])

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

training_sent = to_categorical(np.array(sentiments[0:training_size]), 3) #to_categorical Converts a class vector (integers) to binary class matrix. I have 3 classes
testing_sent = to_categorical(np.array(sentiments[training_size:]), 3)
# training_sent = np.array(sentiments[0:training_size]) #to_categorical Converts a class vector (integers) to binary class matrix. I have 3 classes
# testing_sent = np.array(sentiments[training_size:])

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
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))#output layer = 3 classes POS, NEU, NEG


# from tensorflow.keras import layers, Model, Input

# int_sequences_input = Input(shape=(None,), dtype="int64")
# embedded_sequences = embedding_layer(int_sequences_input)
# x = layers.Conv1D(16, 5, activation="relu")(embedded_sequences)
# x = layers.Dense(16, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
# preds = layers.Dense(3, activation="softmax")(x)
# model = Model(int_sequences_input, preds)

model.compile(
    loss='categorical_crossentropy', #sparse_categorical_crossentropy categorical_crossentropy
    optimizer='Adam', 
    metrics=['accuracy']
)

model.summary()

#--train model
model.fit(training_tweets_seq_pad, training_sent, epochs=5, verbose=1)

#-evaluate model
#pred_train= model.predict(training_tweets_seq_pad)
scores = model.evaluate(training_tweets_seq_pad, training_sent, verbose=1)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
#pred_test= model.predict(testing_tweets_seq_pad)
scores2 = model.evaluate(testing_tweets_seq_pad, testing_sent, verbose=1)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))   