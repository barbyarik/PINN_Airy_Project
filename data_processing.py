# Подключение библиотек и функций
import matplotlib.pyplot as plt
import mplcyberpunk
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

def data_to_tuple(data):
  return [(x, y) for x, y in data.values]

def split_data(dataframe, validation=False):
    if validation:
       train_and_val_dataset, test_dataset =\
        train_test_split(dataframe, test_size=0.2)
       train_dataset, val_dataset =\
        train_test_split(train_and_val_dataset, test_size=0.25)
       return [train_dataset, test_dataset, val_dataset]
    else:
       train_dataset, test_dataset =\
        train_test_split(dataframe, test_size=0.2)
       return [train_dataset, test_dataset]

def mse(Y_true, Y_pred, show=False):
   MSE = mean_squared_error(y_true=Y_true, y_pred=Y_pred)
   if show: print(f"MSE-метрика равна {round(MSE, 5)}")
   return MSE

def rmse(Y_true, Y_pred, show=False):
   RMSE = root_mean_squared_error(y_true=Y_true, y_pred=Y_pred)
   if show: print(f"RMSE-метрика равна {round(RMSE, 5)}")
   return RMSE

def r2(Y_true, Y_pred, show=False):
   R2 = r2_score(y_true=Y_true, y_pred=Y_pred)
   if show: print(f"R2-метрика равна {round(R2, 5)}")
   return R2

def mae(Y_true, Y_pred, show=False):
   MAE = mean_absolute_error(y_true=Y_true, y_pred=Y_pred)
   if show: print(f"MAE-метрика равна {round(MAE, 5)}")
   return MAE

def mape(Y_true, Y_pred, show=False):
   MAPE = mean_absolute_percentage_error(y_true=Y_true, y_pred=Y_pred)
   if show: print(f"MAPE-метрика равна {round(MAPE, 5)}")
   return MAPE

def wape(Y_true, Y_pred, show=False):
   WAPE = sum([abs(true - pred) for true, pred in zip(Y_true, Y_pred)])\
  / sum([abs(y) for y in Y_true])
   if show: print(f"WAPE-метрика равна {round(WAPE, 5)}")
   return WAPE

# Отображение графика y от x
def draw_plot_airy(dataframe_airy, filename='Plot of the Airy function',
                   show=True, save=False):
   X = list(dataframe_airy['X'].values)
   Y = list(dataframe_airy['Y'].values)

   plt.style.use("cyberpunk")
   plt.figure()
   plt.plot(X, Y, label='Численные методы')
   plt.legend(loc='lower right')
   plt.title('Зависимость y от x', weight='bold')
   plt.xlabel('Ось x', weight='bold')
   plt.ylabel('Ось y', weight='bold')
   mplcyberpunk.add_glow_effects()
   if save: plt.savefig(f'images/{filename}.png')
   if show: plt.show()
   
def draw_plot_compare(X, Y, Y_PINN,
                      filename='Comparing PINN and numerical methods',
                      show=True, save=False):
   plt.style.use("cyberpunk")
   plt.figure()
   plt.plot(X, Y, label='Численные методы')
   plt.plot(X, Y_PINN, label='PINN-нейросеть')
   plt.legend(loc='lower right')
   plt.xlabel('Ось x', weight='bold')
   plt.ylabel('Ось y', weight='bold')
   plt.title('Зависимость y от x', weight='bold')
   mplcyberpunk.add_glow_effects()
   if save: plt.savefig(f'images/{filename}.png')
   if show: plt.show()