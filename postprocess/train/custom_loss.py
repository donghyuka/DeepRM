import torch
import torch.nn as nn
import numpy as np


class CustomLossCyclamen(nn.Module):
    def __init__(self, name='cyclamen'):
        super(CustomLossCyclamen, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = torch.mul(y_true, np.pi)
        scaling_factor = torch.sin(scaling_factor)
        scaling_factor = torch.sub(2.0, scaling_factor)
        scaling_factor = torch.div(2.0, scaling_factor)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor)
        loss = torch.mean(loss)
        return loss


class CustomLossFuchsia(nn.Module):
    def __init__(self, name='fuchsia'):
        super(CustomLossFuchsia, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = torch.sub(y_true, 0.5)
        scaling_factor = torch.pow(scaling_factor, 2)
        scaling_factor = torch.mul(scaling_factor, 2.0)
        scaling_factor = torch.sub(1.0, scaling_factor)
        scaling_factor = torch.log(scaling_factor)
        scaling_factor = torch.div(scaling_factor, 2.0 * np.log(2.0))
        scaling_factor = torch.add(1.5, scaling_factor)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor)
        loss = torch.mean(loss)
        return loss


class CustomLossFoxglove(nn.Module):
    def __init__(self, name='foxglove'):
        super(CustomLossFoxglove, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = torch.mul(y_true, np.pi)
        scaling_factor = torch.sin(scaling_factor)
        scaling_factor = torch.sub(2.0, scaling_factor)

        sqerror = torch.pow(torch.sub(y_true, y_pred), 2)
        loss = torch.mul(scaling_factor, sqerror)
        loss = torch.mean(loss)
        return loss


class CustomLossHydrangea(nn.Module):
    def __init__(self, name='hydrangea'):
        super(CustomLossHydrangea, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor_1 = torch.mul(y_true, np.pi)
        scaling_factor_1 = torch.sin(scaling_factor_1)
        scaling_factor_1 = torch.sub(2.0, scaling_factor_1)

        scaling_factor_2 = torch.mul(y_true, np.pi)
        scaling_factor_2 = torch.mul(scaling_factor_2, 2.0)
        scaling_factor_2 = torch.cos(scaling_factor_2)
        scaling_factor_2 = torch.mul(scaling_factor_2, 0.5)
        scaling_factor_2 = torch.sub(1.5, scaling_factor_2)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor_2)
        loss = torch.mul(scaling_factor_1, loss)
        loss = torch.mean(loss)

        return loss



class CustomLossIris(nn.Module):
    def __init__(self, name='iris'):
        super(CustomLossIris, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = torch.sub(y_true, 0.5)
        scaling_factor = torch.pow(scaling_factor, 2)
        scaling_factor = torch.mul(scaling_factor, -4.0)
        scaling_factor = torch.exp(scaling_factor)
        scaling_factor = torch.add(scaling_factor, 1.0)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor)
        loss = torch.mean(loss)
        return loss


class CustomLossSage(nn.Module):
    def __init__(self, name='sage'):
        super(CustomLossSage, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = torch.sub(y_true, 0.5)
        scaling_factor = torch.pow(scaling_factor, 2)
        scaling_factor = torch.mul(scaling_factor, 3.0)
        scaling_factor = torch.sub(1.75, scaling_factor)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor)
        loss = torch.mean(loss)
        return loss




class CustomLossAsparagus(nn.Module):
    def __init__(self, name='asparagus'):
        super(CustomLossAsparagus, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = 1.0 + ((2.0 * y_true - 1.0) ** 4)
        error = torch.abs(y_true - y_pred)

        loss = scaling_factor * error
        loss = torch.mean(loss)
        return loss


class CustomLossBrassica(nn.Module):
    def __init__(self, name='brassica'):
        super(CustomLossBrassica, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = 1.0 + ((2.0 * y_true - 1.0) ** 4)
        error = (y_true - y_pred) ** 2

        loss = scaling_factor * error
        loss = torch.mean(loss)
        return loss



class CustomLossClover(nn.Module):
    def __init__(self, name='clover'):
        super(CustomLossClover, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = 1.0 + ((3.0 * y_true - 1.5) ** 4)
        error = torch.abs(y_true - y_pred)

        loss = scaling_factor * error
        loss = torch.mean(loss)
        return loss



class CustomLossDracena(nn.Module):
    def __init__(self, name='dracena'):
        super(CustomLossDracena, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = 1.0 + ((3.0 * y_true - 1.5) ** 4)
        error = (y_true - y_pred) ** 2

        loss = scaling_factor * error
        loss = torch.mean(loss)
        return loss



class CustomLossEcheveria(nn.Module):
    def __init__(self, name='echeveria'):
        super(CustomLossEcheveria, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):
        scaling_factor = torch.sub(y_true, 0.5)
        scaling_factor = torch.pow(scaling_factor, 2)
        scaling_factor = torch.mul(scaling_factor, 2.0)
        scaling_factor = torch.sub(1.0, scaling_factor)
        scaling_factor = torch.log(scaling_factor)
        scaling_factor = torch.div(scaling_factor,np.log(2.0))
        scaling_factor = torch.add(scaling_factor,2.0)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor)
        loss = torch.mean(loss)
        return loss


class CustomLossFreesia(nn.Module):
    def __init__(self, name='freesia'):
        super(CustomLossFreesia, self).__init__()
        self.name = name


    def forward(self, y_true, y_pred):
        scaling_factor = torch.sub(y_true, 0.5)
        scaling_factor = torch.pow(scaling_factor, 2)
        scaling_factor = torch.mul(scaling_factor, 2.0)
        scaling_factor = torch.sub(1.0, scaling_factor)
        scaling_factor = torch.log(scaling_factor)
        scaling_factor = torch.div(scaling_factor,np.log(2.0))
        scaling_factor = torch.mul(scaling_factor, 3.0)
        scaling_factor = torch.add(scaling_factor,4.0)

        error = torch.sub(y_true, y_pred)
        error = torch.abs(error)

        loss = torch.pow(error, scaling_factor)
        loss = torch.mean(loss)
        return loss
