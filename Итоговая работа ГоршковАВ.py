#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\AVGorshkov\Downloads\HR.csv')
df.head()


# In[2]:


df.info()


# In[3]:


df.isnull().sum()


# In[4]:


'''Рассчитайте основные статистики для переменных
(среднее,медиана,мода,мин/макс,сред.отклонение).'''

df.describe()


# In[5]:


df.mode()


# In[6]:


column_names = df.columns.tolist()

for i in column_names:
    try:
        print("Среднее для ", i, df[i].mean())
        print("Медиана для ", i, df[i].median())
        print("Мода для ", i, df[i].mode()[0])
        print("MIN для ", i, df[i].min())
        print("MAX для ", i, df[i].max())
        print("Среднеквадратическое отклонение для ", i, df[i].std())
    except:
        pass
    


# In[7]:


'''Рассчитайте и визуализировать корреляционную матрицу для
количественных переменных.
Определите две самые скоррелированные и две наименее
скоррелированные переменные.'''

corr_matrix = df[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "left", "promotion_last_5years"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='Reds')


# In[8]:


corr_matrix

corrs = [(corr_matrix.iloc[row,col], df.columns[row], df.columns[col]) for row in range(1, 8) for col in range(row)]
max2 = sorted(corrs, key=lambda x: abs(x[0]), reverse=True)[:2]
min2 = sorted(corrs, key=lambda x: abs(x[0]), reverse=False)[:2]
print("две самые скоррелированные переменные:\n", max2, "\nдве наименее скоррелированные переменные:\n", min2 )


# In[9]:


'''Рассчитайте сколько сотрудников работает в каждом
департаменте.'''

df["department"].value_counts()


# In[10]:


'''Показать распределение сотрудников по зарплатам.'''

sns.countplot(data=df, x="salary")
plt.show()

df.groupby('salary').department.count()


# In[11]:


'''Показать распределение сотрудников по зарплатам в каждом
департаменте по отдельности'''

sns.countplot(data=df, x="department", hue= "salary")
plt.xticks(rotation=90)
plt.show()

df.groupby(['department', 'salary'])\
    .agg({'satisfaction_level': 'count'})\
    .rename(columns={"satisfaction_level": "qty_people"})


# In[12]:


'''Проверить гипотезу, что сотрудники с высоким окладом
проводят на работе больше времени, чем сотрудники с низким
окладом'''

sns.boxplot(data = df, y= 'average_montly_hours', x = 'salary')
plt.show()
df.groupby('salary').agg({'average_montly_hours': ['median', 'min', 'max']}) 


# In[13]:


'''Рассчитать следующие показатели среди уволившихся и не
уволившихся сотрудников (по отдельности):
● Доля сотрудников с повышением за последние 5 лет
● Средняя степень удовлетворенности
● Среднее количество проектов'''

#уволившиеся сотрудники
df_resigned = df[df['left'] == 1]
share_r = df_resigned['promotion_last_5years'].sum()/df_resigned['promotion_last_5years'].count()
mean_rsl = df_resigned['satisfaction_level'].mean()
mean_rnp = df_resigned['number_project'].mean()
print('Доля сотрудников с повышением за последние 5 лет:', f"{share_r:.2%}")
print('Средняя степень удовлетворенности: ', f"{mean_rsl:.2%}")
print('Среднее количество проектов: ', round(mean_rnp, 1))


# In[14]:


#активные сотрудники
df_act = df[df['left'] != 1]
share = df_act['promotion_last_5years'].sum()/df_act['promotion_last_5years'].count()
mean = df_act['satisfaction_level'].mean()
mean_np = df_act['number_project'].mean()
print('Доля сотрудников с повышением за последние 5 лет:', f"{share:.2%}")
print('Средняя степень удовлетворенности: ', f"{mean:.2%}")
print('Среднее количество проектов: ', round(mean_np, 1))


# In[15]:


'''Разделить данные на тестовую и обучающую выборки
Построить модель LDA, предсказывающую уволился ли
сотрудник на основе имеющихся факторов (кроме department и
salary)
Оценить качество модели на тестовой выборки'''


X= df[['satisfaction_level',
 'last_evaluation',
 'number_project',
 'average_montly_hours',
 'time_spend_company',
 'Work_accident',
 'promotion_last_5years']]


# In[16]:


y= df['left']


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[18]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit_transform(X_train,y_train)


# In[19]:


y_pred= lda.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, 
                                y_pred=y_pred, 
                                target_names=["Stay", "Leave"]))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                      display_labels=["Stay", "Leave"])


# In[ ]:




