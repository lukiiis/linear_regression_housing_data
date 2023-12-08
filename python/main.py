import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
housing = pd.read_csv('housing_data.csv')

# Opis danych
# print(housing.info())
# print(housing.describe())
print(housing.head())

# # Wizualizacja rozkładów cech
# housing.hist(bins=20, figsize=(15, 10))
# plt.show()
#
# # Macierz korelacji
# corr_matrix = data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()
#
# # Podział danych na cechy (features) i etykiety (labels)
# X = housing.drop('Price', axis=1)
# y = housing['Price']
#
# # Podział danych na zbiór treningowy i testowy
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Normalizacja danych
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Inicjacja modelu
# model = LinearRegression()
#
# # Trenowanie modelu na danych znormalizowanych
# model.fit(X_train_scaled, y_train)
#
# # Sprawdzenie dokładności modelu na danych testowych
# accuracy = model.score(X_test_scaled, y_test)
# print(f"Dokładność modelu: {accuracy}")
#
# # Przewidywanie cen na zbiorze testowym
# predictions = model.predict(X_test_scaled)
#
# # Obliczenie błędu średniokwadratowego
# mse = mean_squared_error(y_test, predictions)
# print(f"Błąd średniokwadratowy: {mse}")
#
# # Walidacja krzyżowa i grid search dla optymalizacji parametrów
# param_grid = {'normalize': [True, False], 'fit_intercept': [True, False]}
#
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train_scaled, y_train)
#
# best_params = grid_search.best_params_
# print(f"Najlepsze parametry: {best_params}")
#
# # Ustawienie najlepszych parametrów do modelu
# best_model = grid_search.best_estimator_
#
# # Dokładność najlepszego modelu na danych testowych
# best_accuracy = best_model.score(X_test_scaled, y_test)
# print(f"Dokładność najlepszego modelu: {best_accuracy}")