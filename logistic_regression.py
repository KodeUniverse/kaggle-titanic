import pandas as pd
import numpy as np
from data_clean_visualize import data_clean
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# # ----------- PREPROCESSING ----------- #
# train.drop(["PassengerId","Name", "Ticket", "Cabin"], axis=1, inplace=True)
# train = pd.get_dummies(train)
# train.dropna(inplace=True)

# colnames = train.columns
# scaler = MinMaxScaler()

# scaled_train = pd.DataFrame(scaler.fit_transform(train), columns=colnames)

train, scaled_train = data_clean(train)



X_train = scaled_train.drop("Survived", axis=1)
Y_train =  scaled_train.loc[:,"Survived"]
X_test = test

# save PassengerId for adding back to output
passengerIDs = X_test.loc[:,"PassengerId"]

X_test.drop(["PassengerId","Name", "Cabin", "Ticket"], axis=1, inplace=True)
X_test = pd.get_dummies(X_test)
X_test.ffill(inplace=True)
model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
# ----------- OUTPUT ----------- #
y_pred = pd.DataFrame({"PassengerId":passengerIDs, "Survived":y_pred.astype(int)})
y_pred.set_index("PassengerId", inplace=True)
y_pred.to_csv("output/y_pred.csv")