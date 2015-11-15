import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("magic04.data",header=None,delimiter=",")
df.shape
df.head()
df.columns

df.hist()

# Density plots

df.plot(kind='density')
plt.show()

for i in range(0,10):
	df[[i]].plot(kind='density')
	plt.show()


# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

X = df[range(0,10)]
y = df[[10]]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)	

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train.ravel())

pred = clf.predict(X_test)

pd.crosstab(y_test.ravel(),pred,rownames=['actual'],colnames=['pred'])

from sklearn.metrics import f1_score, roc_curve, auc

score = f1_score(y_test,pred,pos_label=['g','h'])

y_test_bin=[1 if x=='g' else 0 for x in y_test]
pred_bin=[1 if x=='g' else 0 for x in pred]

fpr,tpr,thr=roc_curve(y_test_bin,pred_bin)
roc_auc=auc(fpr,tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
