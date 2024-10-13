# Подключение библиотек и функций
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Константы
a = 3  # Правая граница интервала
b = -10  # Левая граница интервала
y0 = 1/(3 ** (2/3) * math.gamma(2/3))  # Начальное значение y
yp0 = -1/(3 ** (1/3) * math.gamma(1/3))  # Начальное значение y'

# Вычисление максимально допустимых шагов для интервалов
def _steps_calculation(a, b, min_data_length, balance_to_zero):
  if balance_to_zero:
    return (abs(b) / min_data_length * 2, a / min_data_length * 2)
  else:
    length = abs(a) + abs(b)
    return (length / min_data_length, length / min_data_length)

# Правая часть системы дифференциальных уравнений второго порядка.
# Args: x: Время, y: Вектор состояния (y[0] = y, y[1] = y').
# Returns: Вектор производных (y' = y[1], y'' = f(x, y)).
def _derivatives(x, y):
    y, yp = y
    ypp = x*y  # y'' = xy
    return [yp, ypp]

def _d_reversed(x, y):
    # Отражаем правую часть уравнения
    x, y = _derivatives(-x, y)
    return [-x, -y]

# Решение задачи Коши с помощью метода Рунге-Кутта 4-го порядка
def get_data_airy(min_data_length=16_500,
                  balance_to_zero=False):
    init_conds = [y0, yp0]  # y(0) = 0.355, y'(0) = -0.259
    x_span_positive = (0, a)  # Промежуток времени для x >= 0
    x_span_negative = (0, -b)  # Промежуток времени для x < 0
    negative_step, positive_step = _steps_calculation(a=a, b=b, # Максимальные шаги
    min_data_length=min_data_length, balance_to_zero=balance_to_zero)

    positive_sol = solve_ivp(_derivatives, x_span_positive,
                            init_conds, method='RK45', max_step=positive_step)
    negative_sol = solve_ivp(_d_reversed, x_span_negative,
                            init_conds, method='RK45', max_step=negative_step)
    
    X = list([-t for t in negative_sol.t][::-1]) + list(positive_sol.t)
    Y = list(negative_sol.y[0][::-1]) + list(positive_sol.y[0])

    dataframe_airy = pd.DataFrame({'X': X, 'Y': Y})

    return dataframe_airy

# Сохранение и загрузка dataframe
def save_dataframe(dataframe, filename='dataframe_airy'):
   dataframe.to_csv(f'dataframes/{filename}.csv', index=False)

def load_dataframe(filename='dataframe_airy'):
   return pd.read_csv(f'dataframes/{filename}.csv')