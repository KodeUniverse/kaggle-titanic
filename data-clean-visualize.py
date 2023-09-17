import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

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

sns.histplot(train, x="Fare", kde=True, ax=axes[1][2])

fig.savefig("output/feature-dist.png")

# One-hot encoding on categorical variables
train = train.drop(["Name", "Cabin", "Ticket"], axis=1)
train = pd.get_dummies(train)
heat_fig, heat_axes = plt.figure(), plt.axes()

# Feature correlation plot
heatmap = sns.heatmap(train.corr(), annot=train.corr(), fmt='.2f', ax=heat_axes)
heat_fig = heatmap.get_figure()
heat_fig.set_size_inches((10,10))
heat_fig.savefig("output/feature-corr.png", dpi=400)









