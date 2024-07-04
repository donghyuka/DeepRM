import torch

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(MLPModel, self).__init__()
        self.activation = torch.nn.ReLU()

        assert num_layers >= 2

        self.layers = []
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        self.layers.append(self.activation)
        self.layers.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers-2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(self.activation)
            self.layers.append(torch.nn.BatchNorm1d(hidden_dim))
            self.layers.append(torch.nn.Dropout(dropout_rate))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*self.layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        return None

    def forward(self, x):
        return self.model(x)



