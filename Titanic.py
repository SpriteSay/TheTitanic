import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# 读取数据
test_df = pd.read_csv("./data/test.csv")
train_df = pd.read_csv("./data/train.csv")

# train_df.head()
# print(train_df.head())
# print(train_df.tail())
# print(train_df.info())

# # 对数值型变量描述
# print(train_df.describe())
# # 对标称型变量描述
# print(train_df.describe(include=['O']))  

# # 查看最有影响的几个值，分别对存活率的影响
# print(train_df[['Sex','Survived']].groupby('Sex').mean())
# print(train_df[['Pclass','Survived']].groupby('Pclass').mean())
# print(train_df[['Age','Survived']].groupby('Age').mean())

# g = sns.FacetGrid(train_df,col='Age')
# g.map(plt.hist,'Age',bins=20)


# 去掉Cabin和Ticket两列数据
train_df = train_df.drop(['Cabin','Ticket','PassengerId'], axis=1)
test_df = test_df.drop(['Cabin','Ticket'], axis=1)
combine = [train_df, test_df]

# 提取姓名中的title，并作为一个新的变量
for data in combine:
    data['Title'] = data.Name.str.extract('([a-zA-Z]+)\.')
# print(pd.crosstab(train_df['Title'],train_df['Sex']))
# 对title进行相应的变换
for data in combine:
    data['Title'] = data['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Major','Rev','Sir'],'Rare')
    data['Title'] = data['Title'].replace(['Mlle','Ms','Lady'],'Miss')
    data['Title'] = data['Title'].replace('Mme','Mrs')
# print(pd.crosstab(train_df['Title'],train_df['Sex']))
# print(train_df[['Title','Survived']].groupby('Title').mean())
# 对title，Sex进行映射，转换成数字表示
title_mapping = {"Mr":1,'Mrs':2,'Miss':3,'Master':4,'Rare':5}
sex_mapping = {'male':0,'female':1}

for data in combine:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    data['Sex'] = data['Sex'].map(sex_mapping)

# print(train_df[['Title','Survived']].groupby('Title').mean())
# print(train_df.head())
# 去除姓名,更新数据
train_df = train_df.drop('Name',axis=1)
test_df = test_df.drop('Name',axis=1)
combine = [train_df, test_df]

# 缺失值填充
# Age缺失填充：根据Pclass和Sex进行区分，对每一种Pclass和Sex下的年龄求中位数，以中位数填充这一类人的Age
median_ages = np.zeros((2,3)) #2Sex,3Pclass
for i in range(2):
    for j in range(3):
        for data in combine:
            age_df = data[(data['Sex'] == i) & (data['Pclass'] == j+1)]['Age'].dropna()
            median_ages[i,j] = (age_df.median()/0.5 + 0.5)*0.5  #【???】
# print(median_ages)
for i in range(2):
    for j in range(3):
        for data in combine:
            data.loc[data.Age.isnull() & (data.Sex == i) & (data.Pclass == j+1) , 'Age'] = median_ages[i,j]
for data in combine:
    data['Age'] = data['Age'].astype(int) 
# print(train_df.head(10))

# 查看年龄切片
train_df['AgeBand'] = pd.cut(data['Age'], 5)
# print(train_df[['AgeBand', 'Survived']].groupby('AgeBand').mean())

# 根据切片，对年龄进行修改
for data in combine:
    data.loc[data['Age']<=16, 'Age'] = 0
    data.loc[(data['Age']>16)&(data['Age']<=32), 'Age'] = 1
    data.loc[(data['Age']>32)&(data['Age']<=48), 'Age'] = 2
    data.loc[(data['Age']>48)&(data['Age']<=64), 'Age'] = 3
    data.loc[data['Age']>64, 'Age'] = 4
# print(train_df.head(10))

# 删除AgeBand属性
train_df = train_df.drop(["AgeBand"], axis = 1)
combine = [train_df, test_df]
# print(train_df.info())

# 将兄弟姐妹合并为家庭人数属性
for data in combine:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
# print(train_df[['FamilySize','Survived']].groupby('FamilySize').mean().sort_values(by = 'Survived',ascending = False))

# 构造IsAlone变量，判断是否是一人
for data in combine:
    data['IsAlone'] = 0
    data.loc[data['FamilySize']==1,'IsAlone'] = 1
# print(train_df[['IsAlone','Survived']].groupby('IsAlone').mean())

# 删除SibSp,Parch属性
train_df = train_df.drop(['SibSp','Parch','FamilySize'], axis = 1)
test_df = test_df.drop(['SibSp','Parch','FamilySize'], axis = 1)
combine = [train_df, test_df]
# print(combine[0].info())
# print(combine[1].info())

# Embark缺失值填充，用最常用值
freq_port = train_df['Embarked'].dropna().mode()[0]
for data in combine:
    data['Embarked'] = data['Embarked'].fillna(freq_port)
# print(train_df[['Embarked','Survived']].groupby('Embarked').mean())
for data in combine:
    data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)

# 对fare进行处理、切片
for data in combine:
    data['Fare'] = data['Fare'].fillna(train_df['Fare'].dropna().median())  
train_df['FareBand'] = pd.qcut(data['Fare'], 4)  #qcut按照数量切片
# print(train_df[['FareBand','Survived']].groupby('FareBand').mean())
for data in combine:
    data.loc[data['Fare'] <= 7.896, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.896)&(data['Fare']<=14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454)&(data['Fare']<=31.472), 'Fare'] = 2
    data.loc[(data['Fare'] > 31.472), 'Fare'] = 2
    data['Fare'] = data['Fare'].astype(int)

train_df = train_df.drop('FareBand', axis=1)
combine = [train_df, test_df]
# print(train_df.head(10))
# print(train_df.info())
# print(test_df.head(10))


# 机器学习方法训练模型
X_train = train_df.drop('Survived', axis = 1)
X_test = test_df.drop('PassengerId', axis = 1)
Y_train = train_df['Survived']
# Y_test = 

# # logistic regression
# logreg = LogisticRegression()
# logreg.fit(X_train,Y_train)
# Y_pred = logreg.predict(X_test)
# print(logreg.score(X_train,Y_train))

# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df['logcoef'] = logreg.coef_[0]
# print(coeff_df)
# print(X_test.shape, Y_pred.shape)
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_LogisticRegression.csv',index = False)


# # SVC: Support vector machine
# svc = SVC()
# svc.fit(X_train,Y_train)
# Y_pred = svc.predict(X_test)
# print(svc.score(X_train,Y_train))
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_SVC.csv',index = False)

# # KNN 
# knn = KNeighborsClassifier()
# knn.fit(X_train,Y_train)
# Y_pred = knn.predict(X_test)
# print('knn:',knn.score(X_train,Y_train))
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_KNN.csv',index = False)

# # GaussianNB
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# print('gaussian:',gaussian.score(X_train,Y_train))
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_GuassianNB.csv',index = False)

# # Perceptron 感知机
# perceptron = Perceptron()
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# print('perceptron:',perceptron.score(X_train,Y_train))
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_Perceptron.csv',index = False)

# # Decision Tree
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# print('decision_tree:',decision_tree.score(X_train,Y_train))
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_Decision_tree.csv',index = False)

# # Random Forest
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# print('random_forest:',random_forest.score(X_train,Y_train))
# pred_df = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':Y_pred})
# pred_df.to_csv('pred_Random_forest.csv',index = False)

