#!/usr/bin/env python
# coding: utf-8

# In[292]:


import numpy as np
import pandas as pd
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score


# In[293]:


dataset = pd.read_csv(r"F:\code\python code implementation\kc2.csv")
X =dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]


# In[ ]:





# In[294]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[295]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[296]:


from sklearn.linear_model import LogisticRegression


# In[297]:


logmodel=LogisticRegression()


# In[298]:


logmodel.fit(X_train,y_train)


# In[299]:


predictions = logmodel.predict(X_test)


# In[300]:


from sklearn.metrics import classification_report


# In[301]:


classification_report(y_test,predictions)


# In[302]:


from sklearn.metrics import confusion_matrix


# In[303]:


confusion_matrix(y_test,predictions)


# In[304]:


from sklearn.metrics import accuracy_score


# In[305]:


accuracy_score(y_test,predictions)


# In[306]:


logmodel.score(X_test,y_test)


# In[307]:


from sklearn.naive_bayes import GaussianNB


# In[308]:


logmodel =GaussianNB()


# In[309]:


logmodel.fit(X_train, y_train)


# In[310]:


predictions = logmodel.predict(X_test)


# In[311]:


from sklearn.metrics import classification_report


# In[312]:


classification_report(y_test,predictions)


# In[313]:


from sklearn.metrics import confusion_matrix


# In[314]:


confusion_matrix(y_test,predictions)


# In[315]:


from sklearn.metrics import accuracy_score


# In[316]:


accuracy_score(y_test,predictions)


# In[317]:


logmodel.score(X_test,y_test)


# In[318]:


from sklearn import svm


# In[319]:


logmodel = svm.SVC()


# In[320]:


logmodel.fit(X_train, y_train)


# In[321]:


predictions = logmodel.predict(X_test)


# In[322]:


from sklearn.metrics import accuracy_score


# In[323]:


accuracy_score(y_test,predictions)


# In[324]:


from sklearn.metrics import classification_report


# In[325]:


classification_report(y_test,predictions)


# In[326]:


from sklearn.metrics import confusion_matrix


# In[327]:


confusion_matrix(y_test,predictions)


# In[ ]:





# In[328]:


from sklearn import tree


# In[329]:


logmodel = tree.DecisionTreeClassifier()


# In[330]:


logmodel.fit(X_train, y_train)


# In[331]:


predictions = logmodel.predict(X_test)


# In[332]:


from sklearn.metrics import accuracy_score


# In[333]:


accuracy_score(y_test,predictions)


# In[334]:


from sklearn.metrics import classification_report


# In[335]:


classification_report(y_test,predictions)


# In[336]:


from sklearn.metrics import confusion_matrix


# In[337]:


confusion_matrix(y_test,predictions)


# In[ ]:





# In[338]:


from sklearn.ensemble import AdaBoostClassifier


# In[339]:


from sklearn.datasets import make_classification


# In[340]:


x_train, y_train = make_classification(n_samples=1000, n_features=4,
n_informative=2, n_redundant=0,
random_state=0, shuffle=False)


# In[341]:


clf = AdaBoostClassifier(n_estimators=100, random_state=0)


# In[342]:


clf.fit(x_train, y_train)


# In[343]:


clf.score(x_train, y_train)


# In[344]:


from sklearn.metrics import classification_report


# In[345]:


classification_report(y_test,predictions)


# In[346]:


from sklearn.metrics import confusion_matrix


# In[347]:


confusion_matrix(y_test,predictions)


# In[400]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[401]:


clf = LinearDiscriminantAnalysis()


# In[402]:


clf.fit(x_train, y_train)


# In[403]:


clf.score(x_train, y_train)


# In[404]:


from sklearn.metrics import classification_report


# In[405]:


classification_report(y_test,predictions)


# In[406]:


from sklearn.metrics import confusion_matrix


# In[407]:


confusion_matrix(y_test,predictions)


# In[356]:


from sklearn import tree


# In[357]:


dest = tree.DecisionTreeClassifier()


# In[358]:


dest = dest.fit(x_train,y_train)


# In[359]:


dest.score(x_train, y_train)


# In[360]:


from sklearn.metrics import classification_report


# In[361]:


classification_report(y_test,predictions)


# In[362]:


from sklearn.metrics import confusion_matrix


# In[363]:


confusion_matrix(y_test,predictions)


# In[364]:


from sklearn.ensemble import GradientBoostingClassifier


# In[365]:


gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.75, max_features=2, max_depth = 2, random_state = 0)


# In[366]:


gb.fit(x_train, y_train)


# In[367]:


gb.score(x_train, y_train)


# In[368]:


from sklearn.metrics import classification_report


# In[369]:


classification_report(y_test,predictions)


# In[370]:


from sklearn.metrics import confusion_matrix


# In[371]:


confusion_matrix(y_test,predictions)


# In[372]:


from sklearn.neural_network import MLPClassifier


# In[373]:


anna = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)


# In[374]:


anna.fit(x_train, y_train)


# In[375]:


anna.score(x_train, y_train)


# In[376]:


from sklearn.metrics import classification_report


# In[377]:


classification_report(y_test,predictions)


# In[378]:


from sklearn.metrics import confusion_matrix


# In[379]:


confusion_matrix(y_test,predictions)


# In[ ]:





# In[380]:


from sklearn.neural_network import BernoulliRBM


# In[381]:


model = BernoulliRBM(n_components=2)


# In[382]:


model.fit(x_train, y_train)


# In[ ]:





# In[383]:


anna.score(x_train, y_train)


# In[392]:


from sklearn.ensemble import BaggingClassifier


# In[393]:


clf = BaggingClassifier()


# In[394]:


clf.fit(x_train, y_train)


# In[395]:


clf.score(x_train, y_train)


# In[396]:


from sklearn.metrics import classification_report


# In[397]:


classification_report(y_test,predictions)


# In[398]:


from sklearn.metrics import confusion_matrix


# In[399]:


confusion_matrix(y_test,predictions)


# In[226]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[227]:


dataset = pd.read_csv(r"F:\code\python code implementation\kc2.csv")
X = dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]


# In[228]:


import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset = pd.read_csv(r"F:\code\python code implementation\kc2.csv")
X = dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]


X = x_train
y = y_train
target_names = y_test

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Dataset')

plt.show()


# In[229]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"F:\code\python code implementation\kc2.csv")
X =dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[230]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[231]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[232]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf = svm.SVC()
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
z_pred = gnb.predict(X_test)
k_pred= clf.predict(X_test)


# In[233]:


from sklearn.metrics import confusion_matrix


# In[234]:


cm = confusion_matrix(y_test, y_pred)


# In[235]:


classifier.score(X_train, y_train)


# In[236]:


gnb.score(X_train, y_train)


# In[237]:


clf.score(X_train, y_train)


# In[238]:


from sklearn.decomposition import PCA
pca = LDA(n_components = 2)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)


# In[239]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf = svm.SVC()
classs = LogisticRegression(random_state = 0)
classs.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
z_pred = gnb.predict(X_test)
k_pred= clf.predict(X_test)


# In[240]:


classs.score(X_train, y_train)


# In[241]:


gnb.score(X_train, y_train)


# In[242]:


clf.score(X_train, y_train)


# In[243]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[244]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
gnb = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)
clf = BaggingClassifier()
classs =BernoulliRBM(n_components=2)
classs.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
z_pred = gnb.predict(X_test)
k_pred= clf.predict(X_test)


# In[245]:


from sklearn.decomposition import KernelPCA


# In[246]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf = svm.SVC()
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
z_pred = gnb.predict(X_test)
k_pred= clf.predict(X_test)


# In[247]:


gnb.score(X_train, y_train)


# In[248]:


classifier.score(X_train,y_train)


# In[249]:


clf.score(X_train,y_train)


# In[250]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
gnb = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)
clf = BaggingClassifier()
classs =BernoulliRBM(n_components=2)
classs.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
z_pred = gnb.predict(X_test)
k_pred= clf.predict(X_test)


# In[251]:


gnb.score(X_train, y_train)


# In[252]:


clf.score(X_train,y_train)


# In[253]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"F:\code\python code implementation\kc2.csv")
X =dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

svc = SVC(random_state=42)
svc.fit(X_train, y_train)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

svc_disp = plot_roc_curve(svc, X_test, y_test)
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=svc_disp.ax_)
rfc_disp.figure_.suptitle("ROC curve comparison")


# In[254]:


from sklearn.neighbors import KNeighborsClassifier


# In[255]:


neigh = KNeighborsClassifier(n_neighbors=3)


# In[256]:


neigh.fit(X_train, y_train)


# In[257]:


clf.score(x_train, y_train)


# In[258]:


from matplotlib.pyplot import figure


# In[259]:


a = ['SVM','Naive bias','Logistic Regression','LDA','Gradient Boosting','ADA boosting','Decision tree']


# In[260]:


b = [87,84,85,95,98,99,100]


# In[261]:


plt.plot(b,a)
plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Classifier Accuracy")
plt.show()


# In[279]:


plt.bar(b,a)
plt.bar(b,a, color='tab:green', linestyle='dashed')

plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Classifier Accuracy")
plt.show()


# In[280]:


c=['ANN','BaggingClassifier']


# In[281]:


d=['69','98']


# In[282]:


d


# In[283]:


import matplotlib.pyplot as plt


# In[284]:


plt.rcParams['figure.figsize']=(6,4)


# In[ ]:





# In[285]:


plt.bar(d,c)
plt.bar(d,c, color='tab:orange', linestyle='dashed')
plt.bar(d, height=0.7, width=0.8, bottom=None,  align='center', data=None)
plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Classifier Accuracy")
plt.show()


# In[269]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['logistic regression', 'SVM', 'naive Bayes']
men_means = [85, 87, 84]
women_means = [89,90,91 ]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width )
rects2 = ax.bar(x + width/2, women_means, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accurecy')
ax.set_title('Comparison of Implemention after LDA')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Bagging Classifier','Neural Network']
men_means = [98, 69]
women_means = [96,87 ]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width )
rects2 = ax.bar(x + width/2, women_means, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accurecy')
ax.set_title('Comparison of Implemention after  KPCA')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[ ]:




