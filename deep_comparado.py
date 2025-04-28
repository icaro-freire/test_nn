#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
import time
from configs.settings import NN

# Configurações para reprodutibilidade e desempenho
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
plt.switch_backend('agg')

def load_and_prepare_data(data_path=os.path.join('data', 'sir_vital.csv'), n_span=400):
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

def setup_device(force_cpu=False):
    """Configura o dispositivo (GPU/CPU)"""
    if force_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    print(f"\n=== Usando dispositivo: {device} ===")
    if device.type == 'cuda':
        print(f"=== Nome da GPU: {torch.cuda.get_device_name(0)} ===")
        print(f"=== Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB ===")
    
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
        torch.backends.cudnn.benchmark = True  # Ativa benchmark do cudnn
        
    return model, optimizer, criterion

def train_model(model, optimizer, criterion, t_train, u_train, epochs=int(200e+3), batch_size=1024):
    """Executa o treinamento com batch processing"""
    loss_history = []
    model.train()
    
    # Criar DataLoader para batch processing
    dataset = torch.utils.data.TensorDataset(t_train, u_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_time = time.time()
    
    for i in tqdm(range(epochs), desc="Treinando"):
        epoch_loss = 0.0
        
        for t_batch, u_batch in dataloader:
            optimizer.zero_grad()
            u_hat = model(t_batch)
            loss = criterion(u_hat, u_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss / len(dataloader))
        
        if (i + 1) % 5000 == 0:
            print(f'Época {i+1}/{epochs} - Loss: {loss_history[-1]:.6f}')
    
    training_time = time.time() - start_time
    print(f'\nTempo total de treinamento: {training_time:.2f} segundos')
    
    return loss_history, training_time

def run_training(device, save_suffix=""):
    """Executa o treinamento completo em um dispositivo específico"""
    print(f"\n=== Iniciando treinamento em {device} ===")
    
    # Carregar dados
    t_data, t_span, u_data, t_data_norm, t_span_norm, u_data_norm, N = load_and_prepare_data()
    
    # Converter para tensores e mover para device
    t_train = torch.tensor(t_data_norm, dtype=torch.float32).view(-1, 1).to(device)
    u_train = torch.tensor(u_data_norm, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_span_norm, dtype=torch.float32).view(-1, 1).to(device)
    
    # Inicializar modelo
    model, optimizer, criterion = initialize_model(device)
    
    # Treinamento
    loss_history, train_time = train_model(model, optimizer, criterion, t_train, u_train)
    
    # Avaliação final
    model.eval()
    with torch.no_grad():
        u_nn_norm = model(t_eval).cpu().numpy()
        u_nn = N * u_nn_norm
    
    # Salvar resultados
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + save_suffix
    
    np.savez(os.path.join(save_dir, f'results_{timestamp}.npz'),
             t_span=t_span,
             u_nn=u_nn,
             loss_history=np.array(loss_history),
             train_time=train_time)
    
    return train_time

def main():
    # Executar em CPU
    cpu_device = setup_device(force_cpu=True)
    cpu_time = run_training(cpu_device, "_CPU")
    
    # Executar em GPU (se disponível)
    if torch.cuda.is_available():
        gpu_device = setup_device()
        gpu_time = run_training(gpu_device, "_GPU")
        
        # Comparação de desempenho
        print(f"\n=== Comparação de Desempenho ===")
        print(f"Tempo CPU: {cpu_time:.2f} segundos")
        print(f"Tempo GPU: {gpu_time:.2f} segundos")
        print(f"Speedup GPU/CPU: {cpu_time/gpu_time:.2f}x")
    else:
        print("\nNenhuma GPU disponível para comparação")

if __name__ == '__main__':
    main()
