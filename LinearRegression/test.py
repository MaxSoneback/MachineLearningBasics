import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "sex", "school"]]
data["sex"] = data["sex"].map({"F": 0, "M": 1})
data["school"] = data["school"].map({"GP": 0, "MS": 1})

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best_acc = float('-inf')

for __ in range(1000):

    # Dela upp vår data-mängd i subarrayer. x_train och y_train är subarrayer av X och y,
    # x_test och y_test testar pricksäkerheten. test_size=0.1 säger att vi använder 10% av
    # ursprungsdatan för att träna vår modell
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Vi gör en superenkel prediktionsmodell, dvs en linjär regression
    lr_model = linear_model.LinearRegression()

    # Vi tränar AKA kalibrerar AKA skapar en regressionslinje baserat på våra sub-arrayer x_train & y_train.
    # Vi gör alltså en regressionslinje (med hjälp av minstakvadratmetoden) baserat på 10% av den totala datamängden
    lr_model.fit(x_train, y_train)

    # acc är förklaringsgraden, alltså R2 för modellen
    acc = lr_model.score(x_test, y_test)
    if acc > best_acc:
        best_acc = acc
        with open('student_model.pickle', 'wb') as pickle_file:
            pickle.dump(lr_model, pickle_file)

print(f"The regression line with the highest accuracy had a coeff. of determination of {best_acc}")
pickle_in = open("student_model.pickle", "rb")
best_model = pickle.load(pickle_in)

# y = kx + m, eller y = k1x1 +k2x2 + k3x3 osv, här är koeff. alltså ki. Intercept är Y-värdet för X=0
print(f"Coefficients: \n{best_model.coef_}")
print(f"Intercept: \n{best_model.intercept_}")

# Nedan matar vi in x-värden in i modellen, varpå modellen m.h.a. regressionslinjen försöker förutsäga värdet på y
predictions = best_model.predict(x_test)

g2_test_vec = np.zeros(len(predictions), dtype=int)
for index, prediction in enumerate(predictions):
    g2_test_vec[index] = x_test[index][1]
    print(f"Prediction: {prediction} \nActual values to predict from: {x_test[index]} \nActual value of prediction: {y_test[index]} \n -----------")


# Plotta faktiska värden av variabeln "p" och "predict" i orange. Blå prickar är predictions

p = "G2"
style.use('ggplot')
pyplot.scatter(data[p], data[predict])
pyplot.scatter(g2_test_vec, predictions, color='blue', alpha=0.7)
print(x_test[1])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()
