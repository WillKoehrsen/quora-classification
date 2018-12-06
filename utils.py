import numpy as np
import json
import keras.backend as K

def load_data(data_name, embedding_name):
    if embedding_name == 'wiki':
        if data_name == 'word':
            embedding_matrix = np.load('word_wiki_embeddings.npy')
    elif embedding_name == 'glove':
        if data_name == 'word':
            embedding_matrix = np.load('word_glove_embeddings.npy')
            
    if data_name == 'word':
        seq_arr = np.load('word_sequences.npy')
        test_seq_arr = np.load('test_word_sequences.npy')
        labels = np.load('word_labels.npy')
        iw = []
        with open('word_index_word.json', 'r') as f:
            for l in f:
                iw.append(json.loads(l))

        index_word = iw[0]
        index_word = {int(key): word for key, word in index_word.items()}

        wi = []
        with open('word_word_index.json', 'r') as f:
            for l in f:
                wi.append(json.loads(l))

        word_index = wi[0]
        word_index = {word: int(index) for word, index in word_index.items()}
            
        vs = len(word_index)
    return seq_arr, test_seq_arr, labels, word_index, index_word, vs, embedding_matrix

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

def format_sequence(s):
    """Add spaces around punctuation."""

    # Add spaces around punctuation
    s = re.sub(
        r'(?<=[^\s])(?=[“”!\"#$%&()*+,./:;<=>?@[\]^_`{|}~\t\n])|(?=[^\s])(?<=[“”!\"#$%&()*+,./:;<=>?@[\]^_`{|}~\t\n])', r' ', s)

    # Remove double spaces
    s = re.sub(r'\s\s', ' ', s)
    return s


def format_data(df_train, df_test,
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                keep_freq=5):
    """Format text data"""
    texts = list(df_train['question_text'])
    texts = [format_sequence(t) for t in texts]

    # Fit once to get word counts
    tokenizer = Tokenizer(lower=False, filters=filters)
    tokenizer.fit_on_texts(texts)
    wc = tokenizer.word_counts
    wc = sorted(wc.items(), key=lambda x: x[1], reverse=True)
    keep = [w for w in wc if w[1] >= keep_freq]

    # Create again to limit to top words
    tokenizer = Tokenizer(num_words=len(keep), 
                          oov_token = 'UNK',
                          lower=False,
                          filters=filters)
    tokenizer.fit_on_texts(texts)
    word_index = dict(list(tokenizer.word_index.items())[:len(keep)])
    index_word = dict(list(tokenizer.index_word.items())[:len(keep)])
    wc = tokenizer.word_counts
    wc = sorted(wc.items(), key=lambda x: x[1], reverse=True)[:len(keep)]
    word_index['PAD'] = 0
    index_word[0] = 'PAD'
    sequences = tokenizer.texts_to_sequences(texts)
    lens = [len(s) for s in sequences]
    
    vs = tokenizer.num_words + 1

    # Pad sequences to have same length
    sequences = pad_sequences(sequences, max(lens))
    sequences = np.array(sequences, dtype=int)

    # Labels
    labels = np.array(df_train['target'], dtype=int)

    # Test data
    test_texts = list(df_test['question_text'])
    test_texts = [format_sequence(t) for t in test_texts]
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_sequences = pad_sequences(test_sequences, max(lens))
    test_sequences = np.array(test_sequences, dtype=int)

    return sequences, labels, test_sequences, word_index, index_word, wc, vs