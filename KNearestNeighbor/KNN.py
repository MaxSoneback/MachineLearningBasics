import pickle
import sklearn
from termcolor import colored
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


def train_sets_fold_split(x_list, y_list, nr_of_folds=5):
    # sklearn har en egen KFold cross validator som både skapar folds + validerar (sklearn.model_selection.KFold),
    # jag försöker här dock konstruera en egen för att få djupare förståelse i ämnet
    x_folds = np.array_split(x_list, nr_of_folds)
    y_folds = np.array_split(y_list, nr_of_folds)
    return x_folds, y_folds


def fold_cross_validation(x_folds, y_folds):
    """
    Målet med fold_cross_validation är att hitta det optimala värdet på k enbart genom att använda vår träningsdata.
    Detta åstadkommer vi först och främst genom att loopa från K = 1 till K = 10, därefter konstruerar vi en inre loop
    där vi loopar över antalet folds 'l' med indexet 'j'. Vi använder sedan fold j som testmängd och resterande folds som
    träningsmängd, s.a. alla folds får agera testmängd. På så vis kan vi testa ett givet k l gånger genom att spara den
    ackumulerade förklaringsgraden för varje varv i den inre loopen. Om genomsnittet på denna förklaringsgrad är högre
    än 'best_avg_acc', spara värdet på k. Testa sedan ett nytt värde på k.
    """
    best_avg_acc = float('-inf')
    best_k = float('-inf')

    for k in range(1, 11):
        cum_acc = 0

        for j in range(0, len(x_folds)):
            x_train, x_test = folds_split_train_and_test(x_folds, j)
            y_train, y_test = folds_split_train_and_test(y_folds, j)
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(x_train, y_train)

            # räkna ut ackumulerad förklaringsgrad, när den inre loopen loopat färdig används detta för att utvärdera
            # hur väl detta k fungerar på datamängden
            cum_acc += model.score(x_test, y_test)

        avg_acc = cum_acc / len(x_folds)
        # print(f'antalet grannar är nu {k} och avg_acc {avg_acc}')

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_k = k
    # print(f'best_k är {best_k}')
    return best_k


def folds_split_train_and_test(folds, index):
    test = folds[index]
    train = np.concatenate(np.delete(folds, index))
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

# Vi vill använda värden i X för att förutsäga y
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(klass)

best_acc = float('-inf')
best_k = 1
for __ in range(10):
    # Vi använder 80% av datamängden som träningsdata, dvs 80% av datan används för att förutsäga resterande 20%.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4)
    x_folds, y_folds = train_sets_fold_split(x_train, y_train)
    k = fold_cross_validation(x_folds, y_folds)
    model = KNeighborsClassifier(n_neighbors=k)

    # Träna modellen
    model.fit(x_train, y_train)

    # Hur väl fungerar den på test-datan?
    acc = model.score(x_test, y_test)

    if acc > best_acc:
        best_acc = acc
        best_k = k
        with open('knn_model.pickle', 'wb') as pickle_file:
            pickle.dump(model, pickle_file)

pickle_in = open("knn_model.pickle", "rb")
best_model = pickle.load(pickle_in)

predictions = best_model.predict(x_test)

names = {0: "unacc", 1: "acc", 2: "good", 3: "vgood"}

for index, prediction in enumerate(predictions):
    color = 'green' if names[prediction] == names[y_test[index]] else 'red'
    print(f"Prediktion: {colored(names[prediction], color)}, Data: {x_test[index]}, Faktiskt värde: {names[y_test[index]]}")

print(f"Modellen med bäst förklaringsgrad hade R2 = {round(best_acc,3)} & k = {best_k}")
