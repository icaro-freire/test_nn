#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
from datetime import datetime
from configs.settings import NN

def load_and_prepare_data(data_path='data/sir_vital.csv', n_span=400):
    """Carrega e prepara os dados"""
    df = pd.read_csv(data_path)
    u_data = df.values
    
    n_data = len(u_data)
    t_min, t_max = 0, n_data - 1
    
    t_data = np.linspace(t_min, t_max, n_data)
    t_span = np.linspace(t_min, t_max, n_span)
    
    # Normalização
    T, N = t_max, u_data[0].sum()
    t_data_norm = t_data / T
    t_span_norm = t_span / T
    u_data_norm = u_data / N
    
    return t_data, t_span, u_data, t_data_norm, t_span_norm, u_data_norm, N

def setup_device():
    """Configura o dispositivo (GPU/CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Usando dispositivo: {device} ===")
    print(f"=== Nome da GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} ===\n")
    return device

def initialize_model(device, input_dim=1, output_dim=3, hidden_dim=16, n_layers=4, activation=nn.Tanh):
    """Inicializa o modelo e componentes de treino"""
    model = NN(input_dim, output_dim, hidden_dim, n_layers, activation).to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    
    # Fixar sementes para reprodutibilidade
    torch.manual_seed(7)
    if device.type == 'cuda':
        torch.cuda.manual_seed(7)
        
    return model, optimizer, criterion

def train_model(model, optimizer, criterion, t_train, u_train, epochs=int(200e+3)):
    """Executa o treinamento"""
    loss_history = []
    model.train()
    
    for i in tqdm(range(epochs), desc="Treinando"):
        optimizer.zero_grad()
        u_hat = model(t_train)
        loss = criterion(u_hat, u_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (i + 1) % 5000 == 0:
            print(f'\nEpoch {i+1}/{epochs} - Loss: {loss.item()}\n')
    
    return loss_history

def save_final_approximation(t_span, u_nn, u_data, save_path):
    """Salva os gráficos da aproximação final"""
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    labels = ['Suscetíveis', 'Infecciosos', 'Removidos']
    
    for i in range(3):
        axs[i].scatter(np.linspace(0, len(u_data)-1, len(u_data)), u_data[:, i], color="tab:orange")
        axs[i].plot(t_span, u_nn[:, i], linewidth=2, color="tab:blue")
        axs[i].set_title(labels[i], fontsize='x-large')
        axs[i].set_xlabel('Tempo', fontsize='large')
    
    axs[0].set_ylabel('Número de Indivíduos', fontsize='large')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results(t_span, u_nn, u_data, loss_history, save_dir='results'):
    """Salva todos os resultados em arquivo"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Salvar dados numéricos
    np.savez(f'{save_dir}/results_{timestamp}.npz',
             t_span=t_span,
             u_nn=u_nn,
             loss_history=np.array(loss_history))
    
    # 2. Salvar gráfico do histórico de loss
    plt.figure(figsize=(10, 4))
    plt.semilogy(loss_history)
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_history_{timestamp}.png')
    plt.close()
    
    # 3. Salvar gráficos da aproximação final (S, I, R)
    save_final_approximation(t_span, u_nn, u_data, f'{save_dir}/final_approximation_{timestamp}.png')

def main():
    # Configuração inicial
    device = setup_device()
    t_data, t_span, u_data, t_data_norm, t_span_norm, u_data_norm, N = load_and_prepare_data()
    
    # Converter para tensores e mover para device
    t_train = torch.tensor(t_data_norm).float().view(-1, 1).to(device)
    u_train = torch.tensor(u_data_norm).float().to(device)
    t_eval = torch.tensor(t_span_norm).float().view(-1, 1).to(device)
    
    # Inicializar modelo
    model, optimizer, criterion = initialize_model(device)
    
    # Treinamento
    loss_history = train_model(model, optimizer, criterion, t_train, u_train)
    
    # Avaliação final
    model.eval()
    with torch.no_grad():
        u_nn_norm = model(t_eval).cpu().numpy()
        u_nn = N * u_nn_norm
    
    # Salvar todos os resultados
    save_results(t_span, u_nn, u_data, loss_history)
    print("\n=== Treinamento concluído com sucesso! ===")
    print(f"=== Resultados salvos no diretório 'results/' ===")
    print(f"=== Arquivos gerados: ===")
    print(f"- results_*.npz: Dados numéricos")
    print(f"- loss_history_*.png: Gráfico do histórico de loss")
    print(f"- final_approximation_*.png: Gráficos S/I/R\n")

if __name__ == '__main__':
    main()
