import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os
from DataPreprocessor import dataprocessor
from model import CustomerReviewModel

DataFrame = pd.read_csv('Dataset/amazon_reviews.csv')

text = DataFrame['text'].tolist()
label = DataFrame['sentiment'].tolist()

vocabulary_size = 1000
embedding_dimension = 20
sequence_length = 75

train_sequence, test_sequence, train_label, test_label = dataprocessor(text, label, vocabulary_size, sequence_length)

model = CustomerReviewModel(vocabulary_size, embedding_dimension, sequence_length)
model.compile(optimizer=Adam(learning_rate=0.01), loss=BinaryCrossentropy(), metrics=[Accuracy()])

earlystopping = EarlyStopping(patience=20)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True)

model.fit(train_sequence, train_label, batch_size=8, epochs=50, callbacks=[earlystopping, model_checkpoint], validation_data=(test_sequence, test_label))
