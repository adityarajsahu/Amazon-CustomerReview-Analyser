from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split


def text_tokenizer(text, vocab_size, seq_len):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    sequences = pad_sequences(sequences, maxlen=seq_len, padding='post', truncating='post')
    return sequences


def dataprocessor(text, label, vocab_size, seq_len):
    label = np.array(label)
    sequence = text_tokenizer(text, vocab_size, seq_len)
    train_sequence, test_sequence, train_label, test_label = train_test_split(sequence, label, test_size=0.3, shuffle=True)
    return train_sequence, test_sequence, train_label, test_label
