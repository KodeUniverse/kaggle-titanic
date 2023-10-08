import pandas as pd
import matplotlib.pyplot as plt
from data_clean_visualize import preprocess
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ----------- PREPROCESSING ----------- #

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

passengerIDs = test.loc[:,"PassengerId"]

train = preprocess(train)
test = preprocess(test)

X_train = train.drop("Survived", axis=1)
Y_train = train.loc[:, "Survived"]
X_test = test

# ----------- HYPERPARAMETERS ----------- #
SEED = 1337
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2
COMPLEXITY_PARAM = 0     #Cost complexity param for post pruning, set to 0 if no post pruning is desired
 
model = DecisionTreeClassifier(criterion = 'entropy',
                               max_depth = MAX_DEPTH, 
                               min_samples_leaf = MIN_SAMPLES_LEAF,
                               ccp_alpha = COMPLEXITY_PARAM,
                               random_state = SEED
                               )

model.fit(X_train, Y_train)
colnames = list(X_train.columns)
# ----------- OUTPUT ----------- #
y_pred = model.predict(X_test)
y_pred = pd.DataFrame({"PassengerId":passengerIDs, "Survived":y_pred.astype(int)})
y_pred.set_index("PassengerId", inplace=True)
y_pred.to_csv("output/decision_tree_pred.csv")

# ----------- TREE PLOTTING ----------- #
fig, axes = plt.subplots(1,1)
fig.set_size_inches(8,8)
fig.set_dpi(2000)
plot_tree(model, ax=axes, label='none', feature_names=colnames)
fig.savefig('output/decision_tree.pdf')

# ----------- TRAINING ACCURACY ----------- #
print(f"Training Accuracy: {accuracy_score(Y_train,model.predict(X_train))}")