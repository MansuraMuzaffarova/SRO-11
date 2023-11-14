# SRO-11
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/XE/Downloads/datasetage_kz.csv")
df = pd.get_dummies(df, columns=['Gender', 'Married'], prefix=['Gender', 'Married'])

# Разделение на признаки (X) и целевую переменную (y)
X = df.drop(columns=['Age', 'Income'])
y = df['Income']
# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Определение моделей
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.05, max_depth=6, alpha=10)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()
# Создание ансамбля моделей
ensemble = VotingRegressor(estimators=[('xgb', xgb_model), ('rf', rf_model), ('lr', lr_model)])
# Обучение ансамбля
ensemble.fit(X_train, y_train)
# Предсказание на тестовом наборе
y_pred = ensemble.predict(X_test)
# Оценка ансамбля
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'R-квадрат: {r2}')
print(f'Средняя абсолютная ошибка (MAE): {mae}')
#Визуализация фактический доход vs. предсказанный доход
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Фактический доход')
plt.ylabel('Предсказанный доход')
plt.title('Фактический vs Предсказанный доход')
plt.show()
