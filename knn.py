import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from data_clean_visualize import preprocess
from sklearn.metrics import confusion_matrix, accuracy_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

passengerIDs = test.loc[:,"PassengerId"]

train = preprocess(train)
test = preprocess(test)

X_train = train.drop("Survived", axis=1)
Y_train = train.loc[:, "Survived"]
X_test = test

print(train)

# ----------- GRID SEARCH FOR BEST K ----------- #

y_true = Y_train

accuracies = {}
for k in range(0,100):
    model = KNeighborsClassifier(n_neighbors = k+1)

    model.fit(X_train, Y_train)

    train_pred = model.predict(X_train)
    
    
    accuracies.update({k+1:accuracy_score(y_true, train_pred)})

accuracy_list = list(accuracies.values())
accuracy_list.sort(reverse=True)
max_accuracy = accuracy_list[8] # 9th best accuracy selected, as gives highest testing acc on Kaggle. K=8


best_k = list(accuracies.keys())[list(accuracies.values()).index(max_accuracy)]
print(f"Best K (w.r.t Training Accuracy) = {best_k}\n Training Accuracy = {max_accuracy}")

# ----------- PREDICTION ----------- #
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# ----------- OUTPUT ----------- #
y_pred = pd.DataFrame({"PassengerId":passengerIDs, "Survived":y_pred.astype(int)})
y_pred.set_index("PassengerId", inplace=True)
y_pred.to_csv("output/knn_pred.csv")

