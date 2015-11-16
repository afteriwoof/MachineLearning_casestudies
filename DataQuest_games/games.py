# Case study from DataQuest.io
# https://www.dataquest.io/blog/getting-started-with-machine-learning-python/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

games = pd.read_csv("games.csv")

print(games.columns.tolist())
print(games.dtypes)
print(games.shape)

headers = games.dtypes.index

games.describe()
games.head()

games.info()

plt.hist(games["average_rating"])
plt.show()

print(games[games["average_rating"]==0].iloc[0])
print(games[games["average_rating"]>0].iloc[0])
# Remove any rows without user reviews
games = games[games["users_rated"]>0]
# And drop rows with missing values
games = games.dropna(axis=0)

#Clustering
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Take only the numeric columns
good_columns = games._get_numeric_data()
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
#Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=labels)
plt.show()

#Finding correlations
games.corr()["average_rating"]

# Filter out the columns we don't want to feed our prediction
columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name"]]

target = "average_rating"

# training/test split
from sklearn.cross_validation import train_test_split
#train = games.sample(frac=0.8,random_state=1)
msk = np.random.rand(len(games))<0.8
train = games[msk]
#test = games.loc[~games.index.isin(train.index)]
test = games[~msk]
print(train.shape)
print(test.shape)

#Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train[columns],train[target])

from sklearn.metrics import mean_squared_error
predictions = model.predict(test[columns])
mean_squared_error(predictions,test[target])

#Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
model.fit(train[columns],train[target])
predictions = model.predict(test[columns])
mean_squared_error(predictions,test[target])


