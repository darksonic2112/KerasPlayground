import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

train_samples = []
train_labels = []
people = {
    "young_total": 1000,
    "old_total": 1000,
    "young_side_effects": 0,
    "old_side_effects": 0
}

plt.xlabel("age")
plt.ylabel("side-effects")

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

"""
for age, effect in zip(train_samples, train_labels):
    print("Patient with the age {age} experienced the effects {effect}".format(age=age, effect=effect))
    if age > 64 and effect:
        people["old_side_effects"] += effect
    else:
        people["young_side_effects"] += effect
    plt.scatter(age, 'side effect' if effect else 'no side effect', color='blue' if effect else 'red')

print("Of {young} young people, {young_side_effects} experienced side-effects.\n"
      "Of {old} old people, {old_side_effects} experienced side-effects".format
      (young=people["young_total"], young_side_effects=people["young_side_effects"], old=people["old_total"],
       old_side_effects=people["old_side_effects"]))

plt.title("Experienced side effects of young and old people")
plt.show()
"""
test_labels = []
test_samples = []

for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))
"""
for i in scaled_train_samples:
    print(i)
"""

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
