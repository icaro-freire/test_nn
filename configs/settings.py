# bibliotecas fundamentais ---------------------------------------------
import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_size, output_size, neurons, hidden_layers, activation):
        super(NN, self).__init__()
        
        # Define a função de ativação
        self.activation = activation
        
        # Cria a lista de camadas
        layers = []
        
        # Adiciona a primeira camada oculta
        layers.append(nn.Linear(input_size, neurons))
        layers.append(self.activation())
        
        # Adiciona as camadas ocultas adicionais
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(self.activation())
        
        # Adiciona a camada de saída
        layers.append(nn.Linear(neurons, output_size))
        
        # Combina todas as camadas em um modelo sequencial
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# função para autodiferenciação dos tensores ---------------------------
def grad(outputs, inputs):
    dFft = torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs = torch.ones_like(outputs), 
        create_graph = True
    )
    return dFft
