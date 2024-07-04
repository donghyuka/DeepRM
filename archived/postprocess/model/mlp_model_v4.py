import torch

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(MLPModel, self).__init__()
        self.activation = torch.nn.ReLU()

        assert num_layers >= 2

        layers_1 = []
        layers_1.append(torch.nn.Linear(input_dim - 7, hidden_dim))
        layers_1.append(self.activation)
        layers_1.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers-2):
            layers_1.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers_1.append(self.activation)
            layers_1.append(torch.nn.Dropout(dropout_rate))
        layers_1.append(torch.nn.Linear(hidden_dim, hidden_dim//4))
        layers_1.append(self.activation)
        self.model_1 = torch.nn.Sequential(*layers_1)

        layers_2 = []
        layers_2.append(torch.nn.Linear(hidden_dim//4 + 1, hidden_dim))
        layers_2.append(self.activation)

        for _ in range(4):
            layers_2.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers_2.append(self.activation)
            layers_1.append(torch.nn.Dropout(dropout_rate))

        layers_2.append(torch.nn.Linear(hidden_dim, hidden_dim // 16))
        layers_2.append(self.activation)
        layers_2.append(torch.nn.Linear(hidden_dim // 16, output_dim))
        layers_2.append(torch.nn.Sigmoid())

        self.model_2 = torch.nn.Sequential(*layers_2)

        self.init_weights()

    def init_weights(self):
        for layer in self.model_1:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        for layer in self.model_2:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        return None

    def forward(self, x):
        pred = self.model_1(x[:,7:])
        pred = torch.cat([pred, x[:,0:1]], dim = 1)
        return self.model_2(pred)
