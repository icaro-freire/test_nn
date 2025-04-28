import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Capacidade CUDA: {torch.cuda.get_device_capability(0)}")
else:
    print("Nenhuma GPU detectada")

# Configurações
input_size = 1000
hidden_size = 500
output_size = 10
num_samples = 10000
batch_size = 128
epochs = 20

# Gerar dados sintéticos
X, y = make_classification(n_samples=num_samples, n_features=input_size, 
                          n_classes=output_size, n_informative=50)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(device_name):
    # Configurar dispositivo
    device = torch.device(device_name)
    print(f"\nTreinando em: {device}")
    
    # Instanciar modelo
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Criar DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Treinamento
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print estatísticas
        epoch_loss = running_loss / len(train_loader)
        print(f"Época {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"Tempo total de treinamento: {training_time:.2f} segundos")
    
    return training_time

# Treinar em CPU
cpu_time = train_model("cpu")

# Treinar em GPU se disponível
if torch.cuda.is_available():
    gpu_time = train_model("cuda")
    print(f"\nRazão de velocidade (GPU/CPU): {gpu_time/cpu_time:.2f}x")
else:
    print("GPU não disponível")
