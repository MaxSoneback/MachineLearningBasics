import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


def train_sets_fold_split(x_list, y_list, nr_of_folds=5):
    # sklearn har en egen KFold cross validator som både skapar folds + validerar (sklearn.model_selection.KFold),
    # jag försöker här dock konstruera en egen för att få djupare förståelse i ämnet
    x_np_array = np.asarray(x_list)
    y_np_array = np.asarray(y_list)
    x_folds = np.array_split(x_np_array, nr_of_folds)
    y_folds = np.array_split(y_np_array, nr_of_folds)
    return x_folds, y_folds


def fold_cross_validation(x_folds, y_folds):
    best_avg_acc = float('-inf')
    best_k = float('-inf')
    for i in range(0,9):
        cum_acc = 0
        for j in range(len(x_folds)):
            x_train, x_test = folds_split_train_and_test(x_folds, index)
            y_train, y_test = folds_split_train_and_test(y_folds, index)

            model = KNeighborsClassifier(n_neighbors=i)
            # Träna modellen
            model.fit(x_train, y_train)

            # Hur väl fungerar den på test-datan?
            cum_acc += model.score(x_test, y_test)

        avg_acc = cum_acc/len(x_folds)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_k = i

        return best_k


def folds_split_train_and_test(folds, index):
    train = [fold for fold_index, fold in enumerate(folds) if fold_index != index]
    test = folds[index]
    return train, test


data = pd.read_csv("car.data")
# print(data.head())

# Vi kan inte utföra operationer på icke-numeriska värden i datamängden. Vi introducerar därför le, där le är ett objekt
# som kan mappa om alla icke-numeriska värden i datamängden till numeriska värden
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
klass = le.fit_transform(list(data["class"]))

predict = "class"

# Vi vill använda X för att se om vi kan använda värden i X för att förutsäga y
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(klass)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

x_folds, y_folds = train_sets_fold_split(x_train, y_train)
k = fold_cross_validation(x_folds, y_folds)

model = KNeighborsClassifier(n_neighbors=9)

# Träna modellen
model.fit(x_train, y_train)

# Hur väl fungerar den på test-datan?
acc = model.score(x_test, y_test)
print(acc)

predictions = model.predict(x_test)

names = {0: "unacc",1: "acc",2: "good",3: "vgood"}

for index, prediction in enumerate(predictions):
    print(f"Predicted: {names[prediction]}, Data: {x_test[index]}, Actual: {names[y_test[index]]}")