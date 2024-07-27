import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import random
import gzip
import os

import subprocess

# Use pip to install the tcn library
subprocess.call(['pip', 'install', 'keras-tcn'])

from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tcn import TCN, tcn_full_summary
from keras.layers import Input, Embedding, Dense, Dropout, SpatialDropout1D, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.utils import to_categorical
from gensim.models import KeyedVectors

# This approach was based on the article: https://towardsdatascience.com/deep-learning-techniques-for-text-classification-78d9dc40bf7c
# It used 

def get_data():
    data = pd.read_table('./train.txt', names = ['label', 'review'])

    # Define a mapping from string labels to integer values
    label_mapping = {
        'TRUTHFULPOSITIVE': 0,
        'TRUTHFULNEGATIVE': 1,
        'DECEPTIVEPOSITIVE': 2,
        'DECEPTIVENEGATIVE': 3
    }

    # Use the map function to replace string labels with integer values
    data['label'] = data['label'].map(label_mapping)
    data.groupby( by='label').count()

    reviews, labels = list(data.review), list(data.label)

    return reviews, labels

def max_length(sequences):
    '''
    input:
        sequences: a 2D list of integer sequences
    output:
        max_length: the max length of the sequences
    '''
    max_length = 0
    for i, seq in enumerate(sequences):
        length = len(seq)
        if max_length < length:
            max_length = length
    return max_length

def data_preprocessing(train_x, test_x):
    # Cleaning and Tokenization
    tokenizer = Tokenizer(oov_token='<UNK>') # this handles punctuation removal, lowering the letter case and tokenization itself
    tokenizer.fit_on_texts(train_x)

    # Turn the text into sequence
    training_sequences = tokenizer.texts_to_sequences(train_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)
    
    max_len = max_length(training_sequences)

    # Pad the sequence to have the same size
    Xtrain = pad_sequences(training_sequences, maxlen=max_len, padding='post', truncating='post')
    Xtest = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

    word_index = tokenizer.word_index
    vocab_size = len(word_index)+1

    return Xtrain, Xtest, max_len, word_index, vocab_size

def define_callbacks():
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, 
                                             patience=10, verbose=2, 
                                             mode='auto', restore_best_weights=True)
    return callbacks

def training_words_in_word2vector(word_to_vec_map, word_to_index):
    '''
    input:
        word_to_vec_map: a word2vec GoogleNews-vectors-negative300.bin model loaded using gensim.models
        word_to_index: word to index mapping from training set
    '''
    
    vocab_size = len(word_to_index) + 1
    count = 0
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        if word in word_to_vec_map:
            count+=1
            
    return print('Found {} words present from {} training vocabulary in the set of pre-trained word vector'.format(count, vocab_size))

def pretrained_embedding_matrix(word_to_vec_map, word_to_index, emb_mean, emb_std):
    '''
    input:
        word_to_vec_map: a word2vec GoogleNews-vectors-negative300.bin model loaded using gensim.models
        word_to_index: word to index mapping from training set
    '''
    np.random.seed(2023) # just the most recent year
    
    # adding 1 to fit Keras embedding (requirement)
    vocab_size = len(word_to_index) + 1
    # define dimensionality of your pre-trained word vectors (= 300)
    emb_dim = 300
    
    # initialize the matrix with generic normal distribution values
    embed_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, emb_dim))
    
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        if word in word_to_vec_map:
            embed_matrix[idx] = word_to_vec_map.get_vector(word)
            
    return embed_matrix

# Had to adapt the original model to be able to whithstand multi-class classification
def tcn_emb_random(kernel_size = 3, activation='relu', input_dim = None, output_dim=300, max_length = None, num_classes=1):
    
    inp = Input( shape=(max_length,))
    x = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length)(inp)
    x = SpatialDropout1D(0.1)(x)
    
    x = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn1')(x)
    x = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn2')(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(num_classes, activation="softmax")(conc) 
    # changed from Dense(1, activation="sigmoid")   

    model = Model(inputs=inp, outputs=outp)
    model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # changed loss function to categorical-crossentropy
    
    return model

# Had to adapt the original model to be able to whithstand multi-class classification
def tcn_word2vec_static(kernel_size = 3, activation='relu', input_dim = None, output_dim=300, max_length=None, num_classes=1, emb_matrix=None):
    
    inp = Input( shape=(max_length,))
    x = Embedding(input_dim=input_dim, 
                  output_dim=output_dim, 
                  input_length=max_length,
                  # Assign the embedding weight with word2vec embedding marix
                  weights = [emb_matrix],
                  # Set the weight to be not trainable (static)
                  trainable = False)(inp)
    
    x = SpatialDropout1D(0.1)(x)
    
    x = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn1')(x)
    x = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn2')(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(num_classes, activation="softmax")(conc) # changed from Dense(1, activation="sigmoid") 

    model = Model(inputs=inp, outputs=outp)
    model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

def t_t_tcn_emb_random():
    # Parameter Initialization
    activations = ['relu', 'tanh']
    kernel_sizes = [1, 2, 3, 4, 5, 6]

    columns = ['Activation', 'Filters', 'acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9', 'acc10', 'AVG']
    record = pd.DataFrame(columns = columns)

    # prepare cross validation with 10 splits and shuffle = True
    kfold = KFold(n_splits=10, shuffle=True)

    # Separate the sentences and the labels
    sentences, labels = get_data()

    exp = 0

    for activation in activations:
        for kernel_size in kernel_sizes:
            exp+=1
            acc_list = []
            for train, test in kfold.split(sentences):
                
                train_x, test_x = [], []
                train_y, test_y = [], []
                
                for i in train:
                    train_x.append(sentences[i])
                    train_y.append(labels[i])

                for i in test:
                    test_x.append(sentences[i])
                    test_y.append(labels[i])

                # Turn the labels into a numpy array
                train_y = np.array(train_y)
                test_y = np.array(test_y)

                # One-hot encode the labels
                train_y = to_categorical(train_y, num_classes=len(np.unique(labels)))
                test_y = to_categorical(test_y, num_classes=len(np.unique(labels)))
                            
                # encode data using
                Xtrain, Xtest, max_length, word_index, vocab_size = data_preprocessing(train_x, test_x)

                # Define the input shape
                model = tcn_emb_random(kernel_size, activation, input_dim=vocab_size, max_length=max_length,  num_classes=len(np.unique(labels)))

                # Train the model
                model.fit(Xtrain, train_y, batch_size=50, epochs=100, verbose=1, 
                        callbacks=[define_callbacks()], validation_data=(Xtest, test_y))

                # evaluate the model
                loss, acc = model.evaluate(Xtest, test_y, verbose=0)
                print('Test Accuracy: {}'.format(acc*100))

                acc_list.append(acc*100)
                
            mean_acc = np.array(acc_list).mean()
            parameters = [activation, kernel_size]
            entries = parameters + acc_list + [mean_acc]

            temp = pd.DataFrame([entries], columns=columns)
            record = pd.concat([record, temp], ignore_index=True)
    
    print("TCN Embedding Random:\n")
    print(record[['Activation', 'AVG']].groupby(by='Activation').max().sort_values(by='AVG', ascending=False))
    return 

def t_t_tcn_word2vec_static():
    compressed_file_path = 'GoogleNews-vectors-negative300.bin.gz'
    decompressed_file_path = 'GoogleNews-vectors-negative300.bin'

    print("Decompressing .gz file...\n")
    with gzip.open(compressed_file_path, 'rb') as compressed_file:
        with open(decompressed_file_path, 'wb') as decompressed_file:
            decompressed_file.write(compressed_file.read())

    print("Loading word2vec...")
    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print("Done!\n")

    print("Removing decompressed file...")
    os.remove(decompressed_file_path)

    # Parameter Initialization
    activations = ['relu']
    emb_mean = word2vec.vectors.mean()
    emb_std = word2vec.vectors.std()
    kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

    columns = ['Activation', 'Filters', 'acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9', 'acc10', 'AVG']
    record = pd.DataFrame(columns = columns)

    # prepare cross validation with 10 splits and shuffle = True
    kfold = KFold(n_splits=10, shuffle=True)

    # Separate the sentences and the labels
    sentences, labels = get_data()

    exp = 0

    for activation in activations:
        for kernel_size in kernel_sizes:
            exp+=1
            print('-------------------------------------------')
            print('Training {}: {} activation, {} kernel size.'.format(exp, activation, kernel_size))
            print('-------------------------------------------')
            
            acc_list = []
            for train, test in kfold.split(sentences):
                
                train_x, test_x = [], []
                train_y, test_y = [], []
                
                for i in train:
                    train_x.append(sentences[i])
                    train_y.append(labels[i])

                for i in test:
                    test_x.append(sentences[i])
                    test_y.append(labels[i])

                # Turn the labels into a numpy array
                train_y = np.array(train_y)
                test_y = np.array(test_y)

                # One-hot encode the labels
                train_y = to_categorical(train_y, num_classes=len(np.unique(labels)))
                test_y = to_categorical(test_y, num_classes=len(np.unique(labels)))
                            
                # encode data using
                Xtrain, Xtest, max_length, word_index, vocab_size = data_preprocessing(train_x, test_x)                
                
                emb_matrix = pretrained_embedding_matrix(word2vec, word_index, emb_mean, emb_std)
                
                # Define the input shape
                model = tcn_word2vec_static(kernel_size, activation, input_dim=vocab_size, max_length=max_length, emb_matrix=emb_matrix, num_classes=len(np.unique(labels)))

                # Train the model
                model.fit(Xtrain, train_y, batch_size=50, epochs=100, verbose=1, callbacks=[define_callbacks()], validation_data=(Xtest, test_y))

                # evaluate the model
                loss, acc = model.evaluate(Xtest, test_y, verbose=0)
                print('Test Accuracy: {}'.format(acc*100))

                acc_list.append(acc*100)
                
            mean_acc = np.array(acc_list).mean()
            parameters = [activation, kernel_size]
            entries = parameters + acc_list + [mean_acc]

            temp = pd.DataFrame([entries], columns=columns)
            record = pd.concat([record, temp], ignore_index=True)

    print("TCN Word2Vec Static:\n")

    return record[['Activation', 'AVG']].groupby(by='Activation').max().sort_values(by='AVG', ascending=False)

t_t_tcn_emb_random()

#t_t_tcn_word2vec_static()