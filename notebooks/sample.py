import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L1
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter
from sklearn.utils import class_weight

#import os
#os.chdir("/Users/davidcsuka/Documents/kdd99_analysis_IDS/notebooks")

train = pd.read_csv("../data/processed/traindata.csv")
test = pd.read_csv("../data/processed/testdata.csv")

encoder = LabelEncoder()
train['labels'] = encoder.fit_transform(train['labels'])
test['labels'] = encoder.fit_transform(test['labels'])

train_x = train.drop("labels", axis=1).values
train_y = train.labels.to_numpy()
test_x = test.drop("labels", axis=1).values
test_y = test.labels.to_numpy()

n_features = train_x.shape[1]

class_weights = dict(zip(np.unique(train_y), class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(train_y),
                                                  y = train_y)))

# define model
model = Sequential()
model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(21, activation='softmax'))
# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# fit the model
model.fit(train_x, train_y, epochs=5, class_weight=class_weights)
#Or
model.fit(train_x, train_y, epochs=5)

loss, acc = model.evaluate(test_x, test_y)

dat = model.predict(test_x)

Counter(list(map(np.argmax, dat)))

train['labels'].value_counts()