import math
import torch
import torch.nn as nn

class SineLayer(nn.Module): #inherits from nn.Module
    
    #in_features: numeber of input neurons, out_features: number of output neurons, omega_0: frequency scaling factor, is_first: if is the first layer
    def __init__(self, in_features=8, out_features, omega_0=30, is_first=False):
        super().__init__() #required parent constructor call
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self): #helps with weight initialization, important for siren stability
        with torch.no_grad(): #no gradient computation for initializing weights, would be inaccurate and unnecessary overhead
            if self.is_first: #small uniform range, uses common heuristic, weight size scales inversly with input dimensionality, can try guassian as experiment in the future?
                self.linear.weight.uniform_(-1/self.in_features, 1/self.in_features) #_ denotes in-place, samples uniformly from interveal [a,b), is stochastic
            else:
                bound = math.sqrt(6/self.in_features)/self.omega_0 #similar to Glorot/Xavier-style factors, divided by omega_0 because if omega_0 is large sin has high density of minima, so weights have to be smaller to avoid oscillations, formula is empiraclly derived from the SIREN paper
                self.linear.weight.uniform_(-bound,bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_() #better for Sirens to not have bias
                
    def forward(self, x):
        return torch.sin(self.omega_0*self.linear(x))                
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0=30.0): #hidden_features: number of neurons in each hidden layer, #hidden_layers:all layers that are not the first or last layer
        super().__init__()
        self.omega_0 = omega_0
        self.net = nn.ModuleList() #a module-only list essentially, and so all parameters are read properly
        
        #first layer
        self.net.append(SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True))
        
        #hidden layers
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
         
        #final layer, so 5 layers total, 4 sin + 1 linear
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad(): #no need to consider gradient of final layer
            self.final_linear.weight.uniform_(-math.sqrt(6/hidden_features)/omega_0, math.sqrt(6/hidden_features)/omega_0) #same logic as in siren layer weight initialization
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()
    
    def forward(self, x):
        for layer in self.net:
            x=layer(x)
        return self.final_linear(x)