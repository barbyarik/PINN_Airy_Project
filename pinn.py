# Подключение библиотек и функций
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from torch.optim import lr_scheduler
from data_airy import y0, yp0, save_dataframe
from data_processing import split_data, data_to_tuple
from data_processing import draw_plot_airy, draw_plot_compare
from data_processing import mse, rmse, r2, mae, mape, wape


def auto_choose_device(verbose=True):
    if verbose:
        print(f"Поддерживается ли CUDA этой системой? {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Будет использоваться GPU")
            print(f"Версия CUDA: {torch.version.cuda}")
            cuda_id = torch.cuda.current_device()
            print(f"ID текущего CUDA устройства: {torch.cuda.current_device()}")
            print(f"Имя текущего CUDA устройства: {torch.cuda.get_device_name(cuda_id)}")
        else:
            device = torch.device('cpu')
            print("Будет использоваться CPU")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return device

# Поскольку в PyTorch не предусмотрено встроенного слоя,
# отвечающего за синус, реализуем его самостоятельно
class SinLayer(nn.Module):
    def __init__(self):
        super(SinLayer, self).__init__()

    def forward(self, x):
        return torch.sin(x)

# Функция потерь для PINN-нейросети в контексте задачи
class MSE_PINN(nn.Module):
  def __init__(self, lamb=1.0, tau=10**(-6)):
    super(MSE_PINN, self).__init__()
    self.lamb, self.tau = lamb, tau

  def forward(self, output, target, boundary_conds, model):
    out0 = model(torch.Tensor([0.]))
    outp0 = (model(torch.Tensor([0. + self.tau])) -\
             model(torch.Tensor([0.]))) / self.tau
    diff = output - target
    diff_boundary_y = (out0 - boundary_conds[0])
    diff_boundary_yp = (outp0 - boundary_conds[1])
    loss = (diff ** 2).mean() +\
           self.lamb*(diff_boundary_y ** 2 + diff_boundary_yp ** 2)
    return loss

# Случайное движение по сетке параметров
def _get_rand_layer(units='need', activation='need'):
  if units == 'need':
    units = random.randint(8, 256)
  if activation == 'need':
    activation = random.choice([nn.Sigmoid(), nn.Tanh(), nn.ReLU(),
                                nn.Softplus(), SinLayer()])
  return (units, activation)

def _get_rand_params(layers='need', epochs='need',
                    batchsize='need', start_lr='need', opt='need'):
  if isinstance(layers, int):
    layers = [_get_rand_layer() for _ in range(layers)]
  if layers == 'need':
    layers = [_get_rand_layer() for _ in range(random.randint(1, 10))]
  if epochs == 'need':
    epochs = random.randint(100, 500)
  if start_lr == 'need':
    start_lr = random.uniform(10**(-3), 10**(-2))
  if batchsize == 'need':
    batchsize = random.choice([16, 32, 64, 128, 256, 512])
  if opt == 'need':
    opt = random.choice(['SGD', 'Adam', 'RMSProp'])
  return (layers, epochs, batchsize, start_lr, opt)

def _compile_model(layers_list, device):
    model = nn.Sequential(*layers_list)
    model = model.to(device=device)
    return model

def _verify_model_architecture(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise ValueError("Optimizer got an empty parameter list")
    
def _opt_setup(opt_name, model, start_lr):
    match opt_name:
        case 'SGD': return torch.optim.SGD(model.parameters(), lr=start_lr)
        case 'Adam': return torch.optim.Adam(model.parameters(), lr=start_lr)
        case 'RMSProp': return torch.optim.RMSprop(model.parameters(), lr=start_lr)

def _fit(model, optimizer, train_loader, epochs, device, criterion, verbose=True):
   scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
   if verbose:
      for epoch in range(1, epochs+1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            # Переносим вычисления на релевантное устройство
            inputs = inputs.to(torch.float32).to(device=device)
            targets = targets.to(torch.float32).to(device=device)

            # Придаем нужную размерность тензору входных и выходных данных
            inputs = inputs.reshape(inputs.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            # Прямое распространение
            scores = model(inputs)
            loss = criterion(scores, targets, [y0, yp0], model)
 
            acc = 1 - mape(Y_true=[t[0] for t in targets.tolist()], 
                           Y_pred=[s[0] for s in scores.tolist()])

            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()

            # Шаг по оптимизатору
            optimizer.step()

            # Обновляем progress bar
            if batch_idx % 5 == 0:
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=loss.item(), acc=acc, lr=scheduler.get_last_lr()[0])

        # Небольшое уменьшение learning_rate
        scheduler.step()
   else:
      for epoch in range(1, epochs+1):
        for (inputs, targets) in train_loader:
            # Переносим вычисления на релевантное устройство
            inputs = inputs.to(torch.float32).to(device=device)
            targets = targets.to(torch.float32).to(device=device)

            # Придаем нужную размерность тензору входных и выходных данных
            inputs = inputs.reshape(inputs.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            # Прямое распространение
            scores = model(inputs)
            loss = criterion(scores, targets, [y0, yp0], model)

            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()

            # Шаг по оптимизатору
            optimizer.step()

        # Небольшое уменьшение learning_rate
        scheduler.step()
   
def save_checkpoint(checkpoint, filename='new_checkpoint', verbose=False):
    if verbose: print('=> Saving checkpoint')
    torch.save(checkpoint, f'checkpoints/{filename}.pth.tar') 

def load_checkpoint(filename, verbose=False):
    if verbose: print('=> Loading checkpoint')
    return torch.load(f'checkpoints/{filename}.pth.tar')

def print_metrics(Y_true, Y_pred):
   mse(Y_true, Y_pred, show=True)
   rmse(Y_true, Y_pred, show=True)
   r2(Y_true, Y_pred, show=True)
   mae(Y_true, Y_pred, show=True)
   mape(Y_true, Y_pred, show=True)
   wape(Y_true, Y_pred, show=True)
   
def random_params_iteration(iteration_num, mse_pinn_lamb, dataframe_airy):
    device = auto_choose_device(verbose=False)
    train, test, val = split_data(dataframe_airy, validation=True)
    X_val = list(val['X'].values)
    Y_true = list(val['Y'].values)
    max_acc = 0

    for _ in range(iteration_num):
        layers, epochs, batchsize, start_lr, opt = _get_rand_params(
            layers='need', epochs='need',
            batchsize='need', start_lr='need', opt='need')
        train_loader = DataLoader(data_to_tuple(train), batch_size=batchsize, shuffle=True)

        layers_list = []
        layers_list += [nn.Linear(1, layers[0][0]), layers[0][1]]
        for i in range(1, len(layers)):
            layers_list += [nn.Linear(layers[i - 1][0], layers[i][0]), layers[i][1]]
        layers_list += [nn.Linear(layers[-1][0], 1)]

        model = _compile_model(layers_list, device)
        criterion = MSE_PINN(lamb=mse_pinn_lamb)
        optimizer = _opt_setup(opt, model, start_lr)
        _verify_model_architecture(model)

        _fit(model, optimizer, train_loader, epochs, device, criterion, verbose=False)

        Y_pred = [float(model(torch.Tensor([x]))) for x in X_val]
        acc_metric = 1 - mape(Y_true, Y_pred)
        if acc_metric > max_acc:
            max_acc = acc_metric
            checkpoint = {'model': model, 'm_state_dict': model.state_dict,
                        'optimizer': optimizer, 'o_state_dict': optimizer.state_dict,
                        'acc_metric': acc_metric}
            save_checkpoint(checkpoint,
                            filename=f'random_launch_chekpoint_{round(acc_metric, 3)}_acc')

def manual_launch(layers_list, epochs, batchsize, start_lr, opt,
                  mse_pinn_lamb, dataframe_airy):
    device = auto_choose_device(verbose=False)
    train, test, val = split_data(dataframe_airy, validation=True)
    X_val = list(val['X'].values)
    Y_val_true = list(val['Y'].values)
    train_loader = DataLoader(data_to_tuple(train), batch_size=batchsize, shuffle=True)
    layers_list, epochs, batchsize, start_lr, opt =\
    layers_list, epochs, batchsize, start_lr, opt

    model = _compile_model(layers_list, device)
    criterion = MSE_PINN(lamb=mse_pinn_lamb)
    optimizer = _opt_setup(opt, model, start_lr)
    _verify_model_architecture(model)

    _fit(model, optimizer, train_loader, epochs, device, criterion, verbose=True)
    
    Y_val_pred = [float(model(torch.Tensor([x]))) for x in X_val]
    acc_metric = 1 - mape(Y_val_true, Y_val_pred)
    checkpoint = {'model': model, 'm_state_dict': model.state_dict,
                'optimizer': optimizer, 'o_state_dict': optimizer.state_dict,
                'acc_metric': acc_metric}
    save_checkpoint(checkpoint, filename=f'manual_launch_chekpoint_{round(acc_metric, 3)}_acc')

    X_test = list(test['X'].values)
    Y_test_true = list(test['Y'].values)
    Y_test_pred = [float(model(torch.Tensor([x]))) for x in X_test]
    print_metrics(Y_test_true, Y_test_pred)
    

    data_tuple = data_to_tuple(dataframe_airy)
    X_all = [elem[0] for elem in data_tuple]
    Y_all_true = [elem[1] for elem in data_tuple]
    Y_all_pred =[float(model(torch.Tensor([x]))) for x in X_all]
    draw_plot_compare(X_all, Y_all_true, Y_all_pred, save=True)

if __name__ == "__main__":
    from data_airy import get_data_airy
    dataframe_airy = get_data_airy()
    save_dataframe(dataframe_airy)

    layers = [
    nn.Linear(1, 50),
    nn.ReLU(),
    nn.Linear(50, 50,),
    SinLayer(),
    nn.Linear(50, 50,),
    SinLayer(),
    nn.Linear(50, 50,),
    nn.ReLU(),
    nn.Linear(50, 1)
    ]
    epochs = 250
    batchsize = 128
    start_lr = 0.001
    opt = 'Adam'

    manual_launch(layers, epochs, batchsize, start_lr, opt,
                  mse_pinn_lamb=0.25, dataframe_airy=dataframe_airy)