import numpy as np
import json
import keras.backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def retrieve_saved_data():
    """Retrieve already formatted data"""
    
    sequences = np.load('sequences.npy')
    test_sequences = np.load('test_sequences.npy')
    labels = np.load('labels.npy')
    
    iw = []
    with open('index_word.json', 'r') as f:
        for l in f:
            iw.append(json.loads(l))

    index_word = iw[0]
    index_word = {int(key): word for key, word in index_word.items()}

    wi = []
    with open('word_index.json', 'r') as f:
        for l in f:
            wi.append(json.loads(l))

    word_index = wi[0]
    word_index = {word: int(index) for word, index in word_index.items()}
    
    vs = len(word_index) + 1
    
    return sequences, labels, test_sequences, word_index, index_word, vs

