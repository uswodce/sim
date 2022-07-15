import math

import torch
import torch.nn as nn

class RandomFourierFeatureLayerCosSin(nn.Module):

    def __init__(self, in_dim, out_dim, kernel, bandwidth):
        super().__init__()

        if out_dim % 2 != 0:
            raise ValueError("This RFF layer requires out_dim is even.")
        half_dim = out_dim // 2

        if kernel == 'gauss':
            W_cos = torch.normal(mean=0.0, std=1./bandwidth, size=(half_dim, in_dim))
            W_sin = W_cos
        elif kernel == 'laplace':
            W_cos = torch.distributions.cauchy.Cauchy(loc=0.0,
                        scale=1./bandwidth).sample([half_dim, in_dim])
            W_sin = W_cos
        else:
            raise ValueError("kernel %s is not supported"%kernel)

        self.linear_cos = nn.Linear(in_dim, half_dim, bias=False) # turn off bias
        self.linear_sin = nn.Linear(in_dim, half_dim, bias=False) # turn off bias
        with torch.no_grad():
            self.linear_cos.weight.copy_(W_cos)
            self.linear_sin.weight.copy_(W_sin)
        self.linear_cos.weight.requires_grad = False # freeze weights
        self.linear_sin.weight.requires_grad = False # freeze weights

        self.norm = torch.sqrt(torch.tensor(half_dim))

    def forward(self, x):
        x_cos = self.linear_cos(x)
        x_sin = self.linear_sin(x)
        z_cos = torch.cos(x_cos)
        z_sin = torch.sin(x_sin)
        z = torch.cat((z_cos, z_sin), dim=-1) / self.norm
        return z


class RandomFourierFeatureLayerCosBias(nn.Module):

    def __init__(self, in_dim, out_dim, kernel, bandwidth):
        super().__init__()

        if kernel == 'gauss':
            W = torch.normal(mean=0.0, std=1./bandwidth, size=(out_dim, in_dim))
        elif kernel == 'laplace':
            W = torch.distributions.cauchy.Cauchy(loc=0.0,
                    scale=1./bandwidth).sample([out_dim, in_dim])
        else:
            raise ValueError("kernel %s is not supported"%kernel)
        b = torch.distributions.uniform.Uniform(torch.tensor([0.0]),
                torch.tensor([2.0 * math.pi])).sample([out_dim])[:,0]

        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.linear.weight.copy_(W)
            self.linear.bias.copy_(b)
        self.linear.weight.requires_grad = False # freeze weights
        self.linear.bias.requires_grad = False # freeze bias

        self.norm = torch.sqrt(torch.tensor(out_dim/2.))

    def forward(self, x):
        z = self.linear(x)
        z = torch.cos(z)
        z = z / self.norm
        return z


class SingleTierFcNet(nn.Module):

    def __init__(self, in_dim=784, out_dim=10, basis_dim=100, tier_dim=100, rev_dim=100):
        super().__init__()

        # first tier
        self.basis = nn.Sequential(
            nn.Linear(in_dim, basis_dim),
            nn.ReLU(),
            nn.Linear(basis_dim, basis_dim),
            nn.ReLU(),
            nn.Linear(basis_dim, tier_dim)
        )

        # dynamics over tier
        self.v = nn.Linear(tier_dim, tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(tier_dim // 2, tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, out_dim)

    def forward(self, x):
        z = self.basis(x)
        z = self.v(z)
        z = self.u(z)
        z = self.inverse_basis(z)
        return self.Wout(z)


class TwoTierFcNet(nn.Module):

    def __init__(self, in_dim=784, out_dim=10, basis_dim=100, first_tier_dim=100,
                    second_tier_dim=100, rev_dim=100, feature='cossin',
                    kernel='gauss', bandwidth=10.0):
        super().__init__()

        # first tier
        self.basis = nn.Sequential(
            nn.Linear(in_dim, basis_dim),
            nn.ReLU(),
            nn.Linear(basis_dim, basis_dim),
            nn.ReLU(),
            nn.Linear(basis_dim, first_tier_dim)
        )

        # second tier
        if feature == 'cossin':
            self.rff = RandomFourierFeatureLayerCosSin(first_tier_dim, second_tier_dim, kernel, bandwidth)
        elif feature == 'cosbias':
            self.rff = RandomFourierFeatureLayerCosBias(first_tier_dim, second_tier_dim, kernel, bandwidth)
        else:
            raise ValueError("feature %s is not supported"%feature)

        # dynamics over tier
        self.v = nn.Linear(second_tier_dim, second_tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(second_tier_dim // 2, second_tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(second_tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, out_dim)

    def forward(self, x):
        z = self.basis(x)
        z = self.rff(z)
        z = self.v(z)
        z = self.u(z)
        z = self.inverse_basis(z)
        return self.Wout(z)


class SingleTierConvNet(nn.Module):

    def __init__(self, in_dim=28, out_dim=10, in_channels=1, out_channels=32,
                    tier_dim=100, rev_dim=100):
        super().__init__()

        # first tier
        # NOTE: pre_dim needs to change if the following basis archtecture changes
        pre_dim = out_channels * (in_dim // 4) ** 2
        self.basis = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(pre_dim, tier_dim)
        )

        # dynamics over tier
        self.v = nn.Linear(tier_dim, tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(tier_dim // 2, tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, out_dim)

    def forward(self, x):
        z = self.basis(x)
        z = self.v(z)
        z = self.u(z)
        z = self.inverse_basis(z)
        return self.Wout(z)

class TwoTierConvNet(nn.Module):

    def __init__(self, in_dim=28, out_dim=10, in_channels=1, out_channels=32, first_tier_dim=100,
                    second_tier_dim=100, rev_dim=100, feature='cossin', kernel='gauss', bandwidth=10.0):
        super().__init__()

        # first tier
        # NOTE: pre_dim needs to change if the following basis archtecture changes
        pre_dim = out_channels * (in_dim // 4) ** 2
        self.basis = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(pre_dim, first_tier_dim)
        )

        # second tier
        if feature == 'cossin':
            self.rff = RandomFourierFeatureLayerCosSin(first_tier_dim, second_tier_dim, kernel, bandwidth)
        elif feature == 'cosbias':
            self.rff = RandomFourierFeatureLayerCosBias(first_tier_dim, second_tier_dim, kernel, bandwidth)

        # dynamics over tier
        self.v = nn.Linear(second_tier_dim, second_tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(second_tier_dim // 2, second_tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(second_tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, out_dim)

    def forward(self, x):
        z = self.basis(x)
        z = self.rff(z)
        z = self.v(z)
        z = self.u(z)
        # z = self.U(z)
        z = self.inverse_basis(z)
        return self.Wout(z)


class SingleTierFourierNet(nn.Module):

    def __init__(self, in_dim=784, out_dim=10, tier_dim=100, rev_dim=100,
                    feature='cossin', kernel='gauss', bandwidth=10.0):
        super().__init__()

        # first tier
        if feature == 'cossin':
            self.rff = RandomFourierFeatureLayerCosSin(in_dim, tier_dim, kernel, bandwidth)
        elif feature == 'cosbias':
            self.rff = RandomFourierFeatureLayerCosBias(in_dim, tier_dim, kernel, bandwidth)
        else:
            raise ValueError("feature %s is not supported"%feature)

        # dynamics over tier
        self.v = nn.Linear(tier_dim, tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(tier_dim // 2, tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, out_dim)

    def forward(self, x):
        z = self.rff(x)
        z = self.v(z)
        z = self.u(z)
        z = self.inverse_basis(z)
        return self.Wout(z)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout_rate=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, channel_sizes, kernel_size=2, dropout_rate=0.2):
        super().__init__()
        layers = []
        num_levels = len(channel_sizes)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else channel_sizes[i-1]
            out_channels = channel_sizes[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout_rate=dropout_rate)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).transpose(1, 2)

class SingleTierTempConvNet(nn.Module):

    def __init__(self, input_size, output_size, channel_sizes, kernel_size, dropout_rate,
                    tier_dim, rev_dim):
        super().__init__()

        # first tier
        self.tcn = TemporalConvNet(input_size, channel_sizes,
                        kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.linear = nn.Linear(channel_sizes[-1], tier_dim)
        self.basis = nn.Sequential(self.tcn, self.linear)

        # dynamics over tier
        self.v = nn.Linear(tier_dim, tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(tier_dim // 2, tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, output_size)

        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.v.weight.data.normal_(0, 0.01)
        self.u.weight.data.normal_(0, 0.01)
        self.inverse_basis[0].weight.data.normal_(0, 0.01)
        self.inverse_basis[2].weight.data.normal_(0, 0.01)
        self.inverse_basis[4].weight.data.normal_(0, 0.01)
        self.Wout.weight.data.normal_(0, 0.01)

    def forward(self, x):
        z = self.basis(x)
        z = self.v(z)
        z = self.u(z)
        z = self.inverse_basis(z)
        return self.Wout(z)

class TwoTierTempConvNet(nn.Module):

    def __init__(self, input_size, output_size, channel_sizes, kernel_size, dropout_rate,
                    first_tier_dim, second_tier_dim, rev_dim,
                    feature='cossin', kernel='gauss', bandwidth=1.0):
        super().__init__()

        # first tier
        self.tcn = TemporalConvNet(input_size, channel_sizes,
                        kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.linear = nn.Linear(channel_sizes[-1], first_tier_dim)
        self.basis = nn.Sequential(self.tcn, self.linear)

        # second tier
        if feature == 'cossin':
            self.rff = RandomFourierFeatureLayerCosSin(first_tier_dim, second_tier_dim, kernel, bandwidth)
        elif feature == 'cosbias':
            self.rff = RandomFourierFeatureLayerCosBias(first_tier_dim, second_tier_dim, kernel, bandwidth)

        # dynamics over tier
        self.v = nn.Linear(second_tier_dim, second_tier_dim // 2, bias=False) # turn off bias
        self.u = nn.Linear(second_tier_dim // 2, second_tier_dim, bias=False) # turn off bias

        # reverse tier
        self.inverse_basis = nn.Sequential(
            nn.Linear(second_tier_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim),
            nn.ReLU(),
            nn.Linear(rev_dim, rev_dim)
        )

        self.Wout = nn.Linear(rev_dim, output_size)

        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.v.weight.data.normal_(0, 0.01)
        self.u.weight.data.normal_(0, 0.01)
        self.inverse_basis[0].weight.data.normal_(0, 0.01)
        self.inverse_basis[2].weight.data.normal_(0, 0.01)
        self.inverse_basis[4].weight.data.normal_(0, 0.01)
        self.Wout.weight.data.normal_(0, 0.01)

    def forward(self, x):
        z = self.basis(x)
        z = self.rff(z)
        z = self.v(z)
        z = self.u(z)
        z = self.inverse_basis(z)
        return self.Wout(z)
