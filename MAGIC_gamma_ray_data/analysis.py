# Case-study on the MAGIC Gamma Telescope Data Set: https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope

import pandas as pd
import matplotlib.pyplot as plt

# Read in the data as comma-separated into a pandas dataframe

df = pd.read_csv("magic04.data",header=None,delimiter=",")
df.shape
df.head()
df.columns

# Density plots

df.hist()

df.plot(kind='density')
plt.show()

# Separate the data by the categorical variable
data_g = df[df[[10]].values=='g']
data_h = df[df[[10]].values=='h']

#for i in range(0,10):
#	df[[i]].plot(kind='density')
#	plt.show()

for i in range(0,10):
	plt.hist(data_g[[i]].values,bins=20,histtype='stepfilled',normed=False,color='b',label='g')
	plt.hist(data_h[[i]].values,bins=20,histtype='stepfilled',normed=False,color='r',alpha=0.5,label='h')
	plt.legend()
	plt.show()


# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

X = df[range(0,10)]
y = df[[10]]
# Split the data into training/test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)	
# Initialise the model
clf = RandomForestClassifier(n_estimators=100)
# Fit the model to the training set
clf.fit(X_train,y_train.ravel())
# Perform a prediction using the model on the test set
pred = clf.predict(X_test)
# Inspect the confusion matrix
pd.crosstab(y_test.ravel(),pred,rownames=['actual'],colnames=['pred'])

from sklearn.metrics import f1_score, roc_curve, auc

# Generate the F1 score
score = f1_score(y_test,pred,pos_label=['g','h'])

# Put the data in binary format for ROC curve
y_test_bin=[1 if x=='g' else 0 for x in y_test]
pred_bin=[1 if x=='g' else 0 for x in pred]
# Determine the ROC curve
fpr,tpr,thr=roc_curve(y_test_bin,pred_bin)
roc_auc=auc(fpr,tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.legend()
plt.show()

