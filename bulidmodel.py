from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

save_name = './result/ROC.csv'

p_train_data = pd.read_csv('./radfeature/A/result_A_train.csv', encoding='gbk')
n_train_data = pd.read_csv('./radfeature/B/result_B_train.csv', encoding='gbk')

train_p = p_train_data.iloc[1:, 1:]
train_n = n_train_data.iloc[1:, 1:]
train_data = np.concatenate((train_p, train_n), axis=1)
train_label = []

feat = p_train_data.iloc[:, 0]

def Zscore(data):
    x_mean = np.mean(data)
    length = len(data)
    vari = np.sqrt((np.sum((data-x_mean)**2))/length)
    data = (data-x_mean)/vari
    return data

def MinMax(data):
    C = data
    min = np.min(C)
    max = np.max(C)
    for one in range(len(data)):
        t = (data[one]-min) / (max-min)
        data[one] = t
    return data

for i in range(0, train_p.shape[1]):
    train_label.append(1)
for i in range(0, train_n.shape[1]):
    train_label.append(0)

x_train = np.array(train_data).transpose()
y_train = np.array(train_label)


p_val_data = pd.read_csv('.', encoding='gbk')
n_val_data = pd.read_csv('.', encoding='gbk')

val_p = p_val_data.iloc[1:, 1:]
val_n = n_val_data.iloc[1:, 1:]
val_data = np.concatenate((val_p, val_n), axis=1)
val_label = []

for i in range(0, val_p.shape[1]):
    val_label.append(1)
for i in range(0, val_n.shape[1]):
    val_label.append(0)

x_val = np.array(val_data).transpose()
y_val = np.array(val_label)

# VIF_list = [variance_inflation_factor(x_val, i) for i in range(x_val.shape[1])]
# print(VIF_list)

# classifier = KNeighborsClassifier(n_neighbors=7)
classifier = RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_depth=10,
                                    min_samples_leaf=7, random_state=123)
# classifier = svm.SVC(probability=True)
# classifier = svm.SVC(probability=True, kernel='rbf', C=10, gamma=1, random_state=123)
# classifier = AdaBoostClassifier(n_estimators=70, learning_rate=0.9, )
probas_ = classifier.fit(x_train, y_train).predict_proba(x_val)
probas_2 = classifier.predict_proba(x_train)
predictions_validation = probas_[:, 1]
y_pre = classifier.predict(x_val)
con = confusion_matrix(y_val, y_pre)

Spe = con[1][1] / (con[0][1] + con[1][1])
Sen = con[0][0] / (con[0][0] + con[1][0])

print(f"SPE{Spe}" + f"\tSEN{Sen}")
acc = accuracy_score(y_val, y_pre)
score = roc_auc_score(y_val, predictions_validation)

print(f'acc:{acc}')
print(f'auc:{score}')
