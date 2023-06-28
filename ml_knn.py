import pandas as pd 

# Шаг 1. Загрузка и очистка данных
df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])

df['Embarked'].fillna('S', inplace = True)
df.drop('Embarked', axis = 1, inplace = True)

age_1 = df[df['Pclass'] == 1]['Age'].median()
age_2 = df[df['Pclass'] == 2]['Age'].median()
age_3 = df[df['Pclass'] == 3]['Age'].median()

def fill_age(row):
   if pd.isnull(row['Age']):
       if row['Pclass'] == 1:
           return age_1
       if row['Pclass'] == 2:
           return age_2
       return age_3
   return row['Age']

df['Age'] = df.apply(fill_age, axis = 1)

def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0

df['Sex'] = df['Sex'].apply(fill_sex)

# Шаг 2. Создание модели
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('Survived', axis = 1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)  #Функция разбивает данные случайным образом на обучающие и тестовые.


sc = StandardScaler()  #Стандартизация значений
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)  #Создать объект классификатора KNN. Настроить параметры модели.
classifier.fit(X_train, y_train)    #«Обучить» модель на тренинговом наборе данных.

y_pred = classifier.predict(X_test) #Рассчитать прогноз значений целевой переменной для тестового набора данных.

print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100) #Оценить точность прогноза. 

print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

TP, TN, FP, FN = 0, 0, 0, 0

#алгоритм распределения по категориям
'''
for test, pred in zip(y_test, y_pred):
    if test - pred == 0:
        if test == 1:
            TP += 1
        else:
            TN += 1
    else:
        if test == 1:
            FN += 1
        else:
            FP += 1   

print('Верный прогноз: выжившие -', TP, 'погибшие -', TN)
print('Ошибочный прогноз: выжившие -', FP, 'погибшие -', FN)

print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
'''
