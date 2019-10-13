import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predictions = model.predict(x_test)

names = {0: "unacc",1: "acc",2: "good",3: "vgood"}

for index, prediction in enumerate(predictions):
    print(f"Predicted: {names[prediction]}, Data: {x_test[index]}, Actual: {names[y_test[index]]}")