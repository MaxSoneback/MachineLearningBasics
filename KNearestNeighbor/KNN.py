import sklearn
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
    ackumulerade accuracyn för varje varv i den inre loopen. om denna accuracy är högre än 'best_avg_acc', spara värdet
    på k. Testa sedan ett nytt värde på k


    """
    best_avg_acc = float('-inf')
    best_k = float('-inf')

    for number_of_neighbors in range(1, 11):
        cum_acc = 0

        for j in range(0, len(x_folds)):
            x_train, x_test = folds_split_train_and_test(x_folds, j)
            y_train, y_test = folds_split_train_and_test(y_folds, j)
            model = KNeighborsClassifier(n_neighbors=number_of_neighbors)
            # Träna modellen
            model.fit(x_train, y_train)

            # Hur väl fungerar den på test-datan?
            cum_acc += model.score(x_test, y_test)

        avg_acc = cum_acc/len(x_folds)
        #print(f'antalet grannar är nu {number_of_neighbors} och avg_acc {avg_acc}')

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_k = number_of_neighbors
    #print(f'best_k är {best_k}')
    return best_k


def folds_split_train_and_test(folds, index):
    #train = [fold for fold_index, fold in enumerate(folds) if fold_index != index]
    test = folds[index]
    train = np.concatenate(np.delete(folds, index))
    #print('modifierad train')
    #print(train)
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
#print(k)
model = KNeighborsClassifier(n_neighbors=k)

# Träna modellen
model.fit(x_train, y_train)

# Hur väl fungerar den på test-datan?
acc = model.score(x_test, y_test)
#print(acc)

predictions = model.predict(x_test)

names = {0: "unacc",1: "acc",2: "good",3: "vgood"}

for index, prediction in enumerate(predictions):
    print(f"Predicted: {names[prediction]}, Data: {x_test[index]}, Actual: {names[y_test[index]]}")