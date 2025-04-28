#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# librarys import ------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import zeros_like
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from configs.settings import NN, grad

# configurações do Qt (para evitar erro de DPI)
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
# configurações do OpenMP (para evitar conflito de DLL)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# setup para usar GPU --------------------------------------
torch.backends.cudnn.benchmark = True
plt.switch_backend('agg')

def setup_device():
    """Configura o dispositivo (GPU/CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\n=== Usando dispositivo: {device} ===")
        print(f"=== Nome da GPU: {torch.cuda.get_device_name(0)} ===")
        print(f"=== Memória disponível: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB ===\n")
    else:
        print("\n=== Usando CPU ===\n")
    return device

# load data ------------------------------------------------
device = setup_device()

# importação dos dados
df = pd.read_csv('data/sir_vital.csv')
u_data = df.values

# gerando malha temporal
n_data = len(u_data)
n_span = 1000

t_min  = 0
t_max  = 99
t_data = np.linspace(t_min, t_max, n_data).astype(int)
t_span = np.linspace(t_min, t_max, n_span)

# normalizando dados 
N = u_data[0].sum()
T = t_max

t_data_norm = t_data / T
t_span_norm = t_span / T
u_data_norm = u_data / N

# transformando em tensores
t_tensor      = torch.tensor(t_data_norm).float().view(-1, 1).to(device)
t_tensor_span = torch.tensor(t_span_norm).float().view(-1, 1).to(device)
u_tensor = torch.tensor(u_data_norm).float().to(device)

# train data 
t_phy = torch.linspace(0, 1, 100, device=device).view(-1, 1).requires_grad_(True)
t_train = t_tensor
u_train = u_tensor

# Criando Dataset e DataLoader (BATCHES DE 32)
dataset = TensorDataset(t_train, u_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # shuffle=False para séries temporais!

# instanciando rede neural ---------------------------------
# hiperparâmetros
lr = 1e-4
epochs = int(90e+3)

eta   = 1/75/365
beta  = 0.25
gamma = 1/12
mu    = eta

eta_nn   = T * eta
beta_nn  = T * beta
gamma_nn = T * gamma
mu_nn    = T * mu

parameters_nn = eta_nn, beta_nn, gamma_nn, mu_nn

# fixando semente aleatória
torch.manual_seed(7)

# instanciando rede neural
model = NN(1, 3, 32, 3, nn.Tanh).to(device)

# acicionando parâmetros para estimação
alpha_1 = torch.rand(1, requires_grad=True, device=device)
alpha_2 = torch.rand(1, requires_grad=True, device=device)
arg_params = [alpha_1, alpha_2]
parameters = list(model.parameters()) + arg_params

# otimizador e métrica
optimizer = Adam(parameters, lr=lr)
normL2 = nn.MSELoss()

# zerando histórico da loss
loss_history = []

# função de treinamento (MODIFICADA PARA BATCHES) ----------
def train_mode():
    """Função para treinamento da PINN com batches"""
    model.train()
    total_loss = 0.0
    total_loss_data = 0.0
    total_loss_residual = 0.0
    
    for t_batch, u_batch in dataloader:
        # Data loss (por batch)
        u_hat = model(t_batch)
        loss_data = normL2(u_hat, u_batch)

        # Residual loss (usa t_phy global)
        u_phy = model(t_phy)
        S_phy = u_phy[:, 0].view(-1, 1)
        I_phy = u_phy[:, 1].view(-1, 1)
        R_phy = u_phy[:, 2].view(-1, 1)

        dSdt = grad(S_phy, t_phy)[0]
        dIdt = grad(I_phy, t_phy)[0]
        dRdt = grad(R_phy, t_phy)[0]

        phy_S = dSdt - (eta_nn - mu_nn * S_phy - beta_nn * S_phy * I_phy)
        phy_I = dIdt - (-mu_nn * I_phy + beta_nn * S_phy * I_phy - gamma_nn * I_phy)
        phy_R = dRdt - (-mu_nn * R_phy + gamma_nn * I_phy)

        loss_S = normL2(phy_S, zeros_like(phy_S))
        loss_I = normL2(phy_I, zeros_like(phy_I))
        loss_R = normL2(phy_R, zeros_like(phy_R))
        loss_residual = loss_S + loss_I + loss_R

        # Total loss (ponderada)
        loss = (alpha_1) * loss_data + (alpha_2) * loss_residual

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumula as losses
        total_loss += loss.item() * len(t_batch)
        total_loss_data += loss_data.item() * len(t_batch)
        total_loss_residual += loss_residual.item() * len(t_batch)
    
    # Médias sobre todos os batches
    avg_loss = total_loss / len(dataset)
    avg_loss_data = total_loss_data / len(dataset)
    avg_loss_residual = total_loss_residual / len(dataset)
    
    return avg_loss, avg_loss_data, avg_loss_residual

# loop de treinamento --------------------------------------
for i in tqdm(range(epochs)):
    # Atualiza parâmetros
    eta_nn, beta_nn, gamma_nn, mu_nn = parameters_nn
    alpha_1, alpha_2 = arg_params
    alpha_1 = torch.exp(alpha_1)
    alpha_2 = torch.exp(alpha_2)
    
    # Forward + Backward
    loss, loss_data, loss_residual = train_mode()
    
    # Guarda no histórico
    loss_history.append(loss)
    
    # Logging
    if (i + 1) % 5000 == 0:
        print(f'''
        ※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※
        loss          = {loss}
        loss_data     = {loss_data}
        loss_residual = {loss_residual}
        ※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※
        alpha_1 = {alpha_1.item()/T}
        alpha_2 = {alpha_2.item()/T}
        ※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※
        ''')

# modo avaliação
model.eval()
with torch.no_grad():
    u_pinn_norm = model(t_tensor_span).cpu().detach().numpy()
    u_pinn = N * u_pinn_norm

# salvando resultados --------------------------------------
def save_results():
    """Função para salvar resultados"""
    save_dir = 'results' 
    os.makedirs(save_dir, exist_ok=True)
    time_mark = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. salvando alphas
    alpha_1_pinn = alpha_1.item() / T
    alpha_2_pinn = alpha_2.item() / T
    np.savetxt(
        os.path.join(save_dir, f'alphas_results_{time_mark}.txt'), 
        [alpha_1_pinn, alpha_2_pinn],
        header="alpha_1, alpha_2"
    )

    # 2. salvando históricos da loss
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, label="Loss History")
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_history_{time_mark}.png'), dpi=300)
    plt.close()

    # 3. salvando gráfico da aproximação final
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    labels = ['Suscetíveis', 'Infecciosos', 'Removidos']

    for i in range(3):
        axs[i].scatter(t_data, u_data[:, i], color="tab:orange", label="Dados Reais")
        axs[i].plot(t_span, u_pinn[:, i], color="tab:red", linewidth=2, label="PINN")
        axs[i].set_title(labels[i], fontsize='x-large')
        axs[i].set_xlabel('Tempo', fontsize='large')
        axs[i].legend(fontsize='large')

    axs[0].set_ylabel('Número de Indivíduos', fontsize='large')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'final_plot_{time_mark}.png'), dpi=300)
    plt.close()

# main section --------------------------------------------
save_results()

print(f'''
≫≫≫≫≫≫≫ FIM DO PROCESSO ≪≪≪≪≪≪≪≪
''')
