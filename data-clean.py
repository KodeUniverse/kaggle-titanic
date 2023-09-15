import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")



train = train.drop(["Name", "Cabin", "Ticket"], axis=1)
train = pd.get_dummies(train)

# Feature correlation plot
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train.corr(), annot=train.corr(), fmt='.2f', ax=ax)
fig.savefig("output/feature-corr.png")








