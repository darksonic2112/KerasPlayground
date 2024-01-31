import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import itertools

train_samples = []
train_labels = []
people = {
    "young_total": 1000,
    "old_total": 1000,
    "young_side_effects": 0,
    "old_side_effects": 0
}

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

for i, j in zip(train_samples, train_labels):
    print("Patient with the age {age} experienced the effects {effect}".format(age=i, effect=j))
    if i > 64 and j:
        people["old_side_effects"] += j
    else:
        people["young_side_effects"] += j

print("Of {young} young people, {young_side_effects} experienced side-effects.\n"
      "Of {old} old people, {old_side_effects} experienced side-effects".format
      (young=people["young_total"], young_side_effects=people["young_side_effects"], old=people["old_total"],
       old_side_effects=people["old_side_effects"]))
