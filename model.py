import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Admission_Predict.csv')
dataset.drop(columns = ['Serial No.', 'LOR ', 'University Rating', 'SOP', 'Research'], inplace = True, axis = 1)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

reg = LinearRegression()
reg.fit(x, y)
pickle.dump(reg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[330, 110, 8.5]]))

