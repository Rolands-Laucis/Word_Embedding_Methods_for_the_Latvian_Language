#this script trains a neural network for text classification on the dataset in ./data into sentiment classes like {positive, negative, neutral...}
#cd sent_nlp
#python train.py --embeddings_path ..\..\Models\fasttext_5_200_sg.txt --embeddings_dim 200 --output_file ..\..\datasets\sentiment\fasttext_5_200_sent.txt

#CODE SNIPPETS CREDIT: https://keras.io/examples/nlp/pretrained_word_embeddings/

#NLP libraries:
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential
#from tensorflow.keras import Input
from tensorflow.keras.layers import InputLayer, Dense, Embedding, Flatten
from tensorflow.keras.metrics import Accuracy, Precision, Recall

#util libs:
import argparse
import numpy as np

#custom scripts and global variables
from parse_raw_data import Parse
#class_names = ["POS", "NEG", "NEU", "IMP", "NOT_LV"]
class_names = ["POS", "NEU", "NEG"]
def F1_Score(precision, recall): return 2*((precision*recall)/(precision+recall))*100

#CLI argument handling
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the .txt word embeddings file.")
parser.add_argument("--embeddings_dim", type=int, required=True, help="Dimension size of word embeddings.")
parser.add_argument("--dataset_file", type=str, default=r'../../datasets/sentiment/data/TweetSetLV.json', help="Path to the dataset .json file.")
parser.add_argument("--verbose", type=bool, default=False, help="Should the program give status updates? Default False.")
parser.add_argument("--gen_output", type=bool, default=True, help="Should the output file with accuracy score be generated? Default True.")
parser.add_argument("--output_file", type=str, default=r'../../datasets/sentiment/results.txt', help="Path to the output .txt file including name of the file.")
args = parser.parse_args()

#--load in tweet text and their sentiments for training data from json (can also be adapted to .csv or .txt; you just need the raw tweet text and the classification labels):
tweets, sentiments = Parse(args.dataset_file, class_names, args.verbose) #my custom script wrapping json and csv file handling in 1 function

#--set up training and testing datasets for text and labels, then init embeddings layer:
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
if args.verbose:
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
if args.verbose:
    print("\nFirst 2 padded sequences of the training set for debug:")
    print(training_tweets_seq_pad[0:2])
    print("All sequences are of length %d" % len(training_tweets_seq_pad[0]))
    print("All sequences are of length %d" % len(testing_tweets_seq_pad[0]))
#clean up memory
del training_tweets_seq
del testing_tweets_seq


#-load in embeddings from local file. Need to create them in a matrix form to use with Keras.
print("\nLoading word embeddings...")
embeddings_index = {}
embedding_dimension = args.embeddings_dim
errors = 0

with open(args.embeddings_path, mode='r', encoding='utf-8') as f:
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
print("Converted %d word vectors from found. (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    input_dim=num_tokens,
    output_dim=embedding_dimension,
    embeddings_initializer= Constant(embedding_matrix),
    trainable=False,
    input_length=len(training_tweets_seq_pad[0])
)

if args.verbose:
    print("Created embedding matrix of shape: %s" % (embedding_matrix.shape,))
#clean up memory
del embeddings_index
del embedding_dimension
del num_tokens
del hits
del misses

#--build model
model = Sequential()
model.add(InputLayer(input_shape=(len(training_tweets_seq_pad[0]),)))#since sequences were padded to 44 length. Can just manually write 44
model.add(embedding_layer)
model.add(Flatten()) #needs to be flattened to get the shapes right
#model.add(Dense(128, activation='relu')) #since SpaCy NLP tasks used 128 nodes
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))#output layer = len(class_names). How many classes there are

model.compile(
    loss='categorical_crossentropy', #sparse_categorical_crossentropy categorical_crossentropy
    optimizer='Adam', 
    metrics=[Accuracy(), Precision(), Recall()] #can just leave the first one to get accuracy. Seems to only care about some metrics during training.
)

if args.verbose:
    model.summary()

#--train model
print("\nTraining model...")
model.fit(training_tweets_seq_pad, training_sent, epochs=12, verbose=1) #SpaCy NLP tasks also used 12 epochs

#-evaluate model
print("\nEvaluating model with metrics %s..." % model.metrics_names)

pred_train = model.predict(training_tweets_seq_pad)
scores = model.evaluate(training_tweets_seq_pad, training_sent, verbose=0)
print('Accuracy on training data: %.2f\nError on training data: %.2f\nF-1 score: %.2f' % (scores[1]*100, (1 - scores[1])*100, F1_Score(scores[2], scores[3])))  
if args.verbose:
    print("Predictions for first few training data tweets")
    for i in range(len(first_train_tweets)):
        print("\n%s\n%s\n%s" % (first_train_tweets[i],class_names[np.argmax(pred_train[i])], pred_train[i])) 
 
pred_test= model.predict(testing_tweets_seq_pad)
scores2 = model.evaluate(testing_tweets_seq_pad, testing_sent, verbose=0)
print('Accuracy on test data: %.2f\nError on training data: %.2f\nF-1 score: %.2f' % (scores2[1]*100, (1 - scores2[1])*100, F1_Score(scores2[2], scores2[3]))) 
if args.verbose:
    print("Predictions for first few test data tweets")
    for i in range(len(first_test_tweets)):
        print("\n%s\n%s\n%s" % (first_test_tweets[i],class_names[np.argmax(pred_test[i])], pred_test[i])) 

#--Save the results to a local file
if args.gen_output:
    with open(args.output_file, mode='w', encoding='utf-8') as output:
        #output.write('Accuracy (%%) on train data, then next line - Error on test data\n%.6f\n%.6f\n\n' % (scores[1]*100, (1 - scores[1])*100))
        output.write('Accuracy (%%) on test data; Error on test data; F1-score\n%.6f\n%.6f\n%.6f' % (scores2[1]*100, (1 - scores2[1])*100, F1_Score(scores2[2], scores2[3])))
    print("\nSaved results to file.")
