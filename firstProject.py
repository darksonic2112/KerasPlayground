import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.pyplot as plt

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
