import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense


class CustomerReviewModel(tf.keras.Model):

    def __init__(self, vocab_size, embed_dim, seq_len):
        super(CustomerReviewModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
    
        self.embedding = Embedding(self.vocab_size, self.embed_dim, input_length=self.seq_len)
        self.lstm1 = Bidirectional(LSTM(self.embed_dim, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(self.embed_dim, return_sequences=True))
        self.lstm3 = Bidirectional(LSTM(self.embed_dim, return_sequences=False))
        self.dense1 = Dense(8, activation='tanh')
        self.dense2 = Dense(2, activation='tanh')
        self.dense3 = Dense(1, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
