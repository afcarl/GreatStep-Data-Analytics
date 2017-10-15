# coding: utf-8

# In[62]:

import pandas as pd

from sklearn.model_selection import StratifiedKFold as SKF
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier as DT

import matplotlib.pyplot as plt


training = pd.read_csv('cust_data.csv')

state_dict = sorted(training['state'].unique())
state_mapping = dict(zip(state_dict, range(0, len(state_dict) + 1)))
training['state'] = training['state'] \
                           .map(state_mapping) \
                           .astype(int)

area_code_dict = sorted(training['area_code'].unique())
area_code_mapping = dict(zip(area_code_dict, range(0, len(area_code_dict) + 1)))
training['area_code'] = training['area_code'] \
                           .map(area_code_mapping) \
                           .astype(int)


international_plan_dict = sorted(training['international_plan'].unique())
international_plan_mapping = dict(zip(international_plan_dict, range(0, len(international_plan_dict) + 1)))
training['international_plan'] = training['international_plan'] \
                           .map(international_plan_mapping) \
                           .astype(int)


voice_mail_plan_dict = sorted(training['voice_mail_plan'].unique())
voice_mail_plan_mapping = dict(zip(voice_mail_plan_dict, range(0, len(voice_mail_plan_dict) + 1)))
training['voice_mail_plan'] = training['voice_mail_plan'] \
                           .map(voice_mail_plan_mapping) \
                           .astype(int)


churn_dict = sorted(training['churn'].unique())
churn_mapping = dict(zip(churn_dict, range(0, len(churn_dict) + 1)))
training['churn'] = training['churn'] \
                           .map(churn_mapping) \
                           .astype(int)

y = training['churn']
y = np.asarray(y)

del training['churn']
del training['phone_number']
del training['Id']


X = training.as_matrix()

t = training.columns.values.tolist()
from xgboost import *
XGBC = XGBClassifier

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

clf_class= XGBC # Replace the classifier here
to_add = 'XGBoost Classifier' # Replace the name here
# kwargs 
# Construct a kfolds object
skf = SKF(n_splits=5,shuffle=True)
skf.get_n_splits(X, y)
y_pred = y.copy()
flag = False
# Iterate through folds
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train = y[train_index]
    # Initialize a classifier with key word arguments
    clf = clf_class()
    clf.fit(X_train,y_train)
    y_pred[test_index] = clf.predict(X_test)
    # if (not flag):
    #   flag = True
    #   feature_importance = clf.feature_importances_
    #   # make importances relative to max importance
    #   feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #   sorted_idx = np.argsort(feature_importance)
    #   pos = np.arange(sorted_idx.shape[0]) + .5
    #   plt.barh(pos, feature_importance[sorted_idx], align='center')
    #   temp  = [t[i] for i in sorted_idx]
    #   plt.yticks(pos, temp)
    #   plt.xlabel('Relative Importance')
    #   plt.title('Variable Importance from Decision Tree:')
    #   plt.show()


np.mean(y == y_pred)
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print (to_add, ':')
print ('Accuracy: ', accuracy_score(y, y_pred))
print ('Precision: ', precision_score(y, y_pred))
print ('Recall: ', recall_score(y, y_pred))
print ('F1: ', f1_score(y, y_pred))


cnf_matrix = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False', 'True'],
                      title=to_add + ': Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False', 'True'], normalize=True,
                      title=to_add + ': Normalized confusion matrix')

plt.show()






del training['total_day_charge']
del training['total_intl_charge']
del training['total_eve_charge']
del training['total_night_charge']
del training['area_code']
X = training.as_matrix()

t = training.columns.values.tolist()
from xgboost import *
XGBC = XGBClassifier

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

clf_class= XGBC # Replace the classifier here
to_add = 'XGBoost Classifier' # Replace the name here
# kwargs 
# Construct a kfolds object
skf = SKF(n_splits=5,shuffle=True)
skf.get_n_splits(X, y)
y_pred = y.copy()
flag = False
# Iterate through folds
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train = y[train_index]
    # Initialize a classifier with key word arguments
    clf = clf_class()
    clf.fit(X_train,y_train)
    y_pred[test_index] = clf.predict(X_test)
    if (not flag):
      flag = True
      feature_importance = clf.feature_importances_
      # make importances relative to max importance
      feature_importance = 100.0 * (feature_importance / feature_importance.max())
      sorted_idx = np.argsort(feature_importance)
      pos = np.arange(sorted_idx.shape[0]) + .5
      plt.barh(pos, feature_importance[sorted_idx], align='center')
      temp  = [t[i] for i in sorted_idx]
      plt.yticks(pos, temp)
      plt.xlabel('Relative Importance')
      plt.title('Variable Importance from '+ to_add)
      plt.show()


np.mean(y == y_pred)
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print (to_add, ':')
print ('Accuracy: ', accuracy_score(y, y_pred))
print ('Precision: ', precision_score(y, y_pred))
print ('Recall: ', recall_score(y, y_pred))
print ('F1: ', f1_score(y, y_pred))


cnf_matrix = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False', 'True'],
                      title=to_add + ': Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False', 'True'], normalize=True,
                      title=to_add + ': Normalized confusion matrix')

plt.show()





X = training.as_matrix()

t = training.columns.values.tolist()
from xgboost import *
XGBC = XGBClassifier

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

clf_class= XGBC # Replace the classifier here
to_add = 'XGBoost Classifier' # Replace the name here
# kwargs 
# Construct a kfolds object
def param_selector(**kwargs):
  skf = SKF(n_splits=5,shuffle=True)
  skf.get_n_splits(X, y)
  y_pred = y.copy()
  flag = False
  # Iterate through folds
  for train_index, test_index in skf.split(X, y):
      X_train, X_test = X[train_index], X[test_index]
      y_train = y[train_index]
      # Initialize a classifier with key word arguments
      clf = clf_class(**kwargs)
      clf.fit(X_train,y_train)
      y_pred[test_index] = clf.predict(X_test)
  return y_pred


y_pred = param_selector(n_estimators = 200)
np.mean(y == y_pred)
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print (to_add, ':')
print ('Accuracy: ', accuracy_score(y, y_pred))
print ('Precision: ', precision_score(y, y_pred))
print ('Recall: ', recall_score(y, y_pred))
print ('F1: ', f1_score(y, y_pred))

for i in [200, 300, 400, 500, 600, 700, 800, 900]:
  y_pred = param_selector(n_estimators = i)
  print (i, precision_score(y, y_pred), accuracy_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred))

for i in [700, 800, 900]:
  for j in [3,4,5, 6]:
    y_pred = param_selector(n_estimators = i, max_depth = j)
    print (i,j, precision_score(y, y_pred), accuracy_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred))



cnf_matrix = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False', 'True'],
                      title=to_add + ': Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False', 'True'], normalize=True,
                      title=to_add + ': Normalized confusion matrix')

plt.show()

