import pandas as pd
import numpy as np
from data_clean_visualize import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# save PassengerId for adding back to output
passengerIDs = test.loc[:,"PassengerId"]

y_true = train.loc[:,"Survived"]

train = preprocess(train)
test = preprocess(test)

colnames = train.columns

# ----------- SCALING ----------- #
scaler = MinMaxScaler()
Y_train = train.loc[:,"Survived"]
X_train = pd.DataFrame(scaler.fit_transform(train.drop("Survived", axis=1)), columns=colnames[1:])
X_test = pd.DataFrame(test, columns=colnames[1:])

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
# ----------- OUTPUT ----------- #
y_pred = pd.DataFrame({"PassengerId":passengerIDs, "Survived":y_pred.astype(int)})
y_pred.set_index("PassengerId", inplace=True)
y_pred.to_csv("output/y_pred.csv")


# ----------- TRAINING ACCURACY ----------- #
print(f"{confusion_matrix(y_true[:418], y_pred)}\nTraining Accuracy: {accuracy_score(y_true[:418], y_pred)}")
