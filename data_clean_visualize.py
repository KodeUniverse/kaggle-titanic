import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler


def preprocess(df):

    df.ffill(inplace=True)
    alone_col = np.where(df["SibSp"] + df["Parch"] > 0, 1, 0)
    df.drop(["PassengerId","Name", "Ticket", "Cabin", "SibSp", "Parch"], axis=1, inplace=True)
    df["TraveledAlone"] = alone_col
    df = pd.get_dummies(df)
    
    colnames = list(df.columns)
    return pd.DataFrame(df, columns=colnames)


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Data Shapes

print(f"Training Data Shape: {train.shape} \n Testing Data Shape: {test.shape}")
"""
GRAPHS AND PLOTS
"""

# Feature distributions

# want to plot dist of features and countplots
fig, axes = plt.subplots(2,3) # 2 x 3 matrix of axes
fig.set_size_inches(10,8)
fig.tight_layout(pad=2.5)
# Count plots
for ind, val in enumerate(["Sex","Survived", "Pclass"]):
    sns.countplot(train, x=val, ax=axes[0][ind])

sns.histplot(train, x="Age", kde=True, ax=axes[1][0])

sns.countplot(train, x="Survived", hue="Sex", ax=axes[1][1])


train = preprocess(train)
sns.countplot(train, x="TraveledAlone", ax=axes[1][2]) # Feature TraveledAlone is only created after preprocess() call

fig.savefig("output/feature-dist.png")

# One-hot encoding on categorical variables
#train = train.drop(["Name", "Cabin", "Ticket"], axis=1)
#train = pd.get_dummies(train)
heat_fig, heat_axes = plt.figure(), plt.axes()

# Feature correlation plot
heatmap = sns.heatmap(train.corr(), annot=np.array(train.corr()), fmt='.2f', cmap='hot', ax=heat_axes)
heat_fig = heatmap.get_figure()
heat_fig.set_size_inches((10,10))
heat_fig.savefig("output/feature-corr.png", dpi=400)
print(np.array(train.corr()))

