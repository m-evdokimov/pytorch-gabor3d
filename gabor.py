import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dGabor(nn.Module):
    '''
    Applies a 3d convolution over an input signal using Gabor filter banks.
    WARNING: the size of the kernel must be an odd number otherwise it'll be shifted with respect to the origin
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 size: int,
                 padding=None,
                 device='cpu'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = in_channels * out_channels
        self.size = size
        self.device = device

        if padding:
            self.padding = padding
        else:
            self.padding = 0

        # all additional axes are made for correct broadcast
        # the bounds of uniform distribution adjust manually for every size (rn they're adjusted for 5x5x5 filters)
        # for better understanding: https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97
        self.sigma = nn.Parameter(torch.Tensor(size=(self.num_filters, 1, 1, 1)).uniform_(2, 10))

        self.thetas = nn.Parameter(torch.Tensor(size=(self.num_filters, 3)).uniform_(-math.pi, math.pi))

        self.gamma_y = nn.Parameter(torch.Tensor(size=(self.num_filters, 1, 1, 1)).uniform_(0.5, 3.5))
        self.gamma_z = nn.Parameter(torch.Tensor(size=(self.num_filters, 1, 1, 1)).uniform_(0.5, 3.5))

        self.lambd = nn.Parameter(torch.Tensor(size=(self.num_filters, 1, 1, 1)).uniform_(4, 10))
        self.psi = nn.Parameter(torch.Tensor(size=(self.num_filters, 1, 1, 1)).uniform_(-math.pi / 4, math.pi / 4))

        self.conv = F.conv3d

    def forward(self, data):
        ''' Input: torch.Tensor with size (B, C, D, H, W) '''
        return self.conv(input=data, weight=self.init_kernel(), padding=self.padding)

    def init_kernel(self):
        '''
        Initialize a gabor kernel with given parameters
        Returns torch.Tensor with size (out_channels, in_channels, size, size, size)
        '''
        lambd = self.lambd
        psi = self.psi

        sigma_x = self.sigma
        sigma_y = self.sigma * self.gamma_y
        sigma_z = self.sigma * self.gamma_z
        R = self.get_rotation_matrix().reshape(self.num_filters, 3, 3, 1, 1, 1)

        c_max, c_min = int(self.size / 2), -int(self.size / 2)
        (z, y, x) = torch.meshgrid(torch.arange(c_min, c_max + 1), torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1))

        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        # meshgrid for every filter
        z = z.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)
        y = y.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)
        x = x.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)

        z_prime = z * R[:, 0, 0] + y * R[:, 0, 1] + x * R[:, 0, 2]
        y_prime = z * R[:, 1, 0] + y * R[:, 1, 1] + x * R[:, 1, 2]
        x_prime = z * R[:, 2, 0] + y * R[:, 2, 1] + x * R[:, 2, 2]

        # gabor formula
        kernel = torch.exp(-.5 * (x_prime ** 2 / sigma_x ** 2 + y_prime ** 2 / sigma_y ** 2 + z_prime ** 2 / sigma_z ** 2)) \
                 * torch.cos(2 * math.pi * x_prime / (lambd + 1e-6) + psi)

        return kernel.reshape(self.out_channels, self.in_channels, self.size, self.size, self.size)

    def get_rotation_matrix(self):
        '''
        Makes 3d rotation matrix.
        In simplest case with one filter it goes:
            R_z = torch.Tensor([[cos_a, -sin_a, 0],
                               [sin_a, cos_a,  0],
                               [0,     0,      1]],)

            R_y = torch.Tensor([[cos_b,  0, sin_b],
                               [0    ,  1,    0],
                               [-sin_b, 0, cos_b]])

            R_x = torch.Tensor([[1,  0,     0],
                               [0,  cos_g, -sin_g],
                               [0,  sin_g, cos_g]])
        but after such definition thetas lose the gradients
        '''
        sin_a, cos_a = torch.sin(self.thetas[:, 0]), torch.cos(self.thetas[:, 0])
        sin_b, cos_b = torch.sin(self.thetas[:, 1]), torch.cos(self.thetas[:, 1])
        sin_g, cos_g = torch.sin(self.thetas[:, 2]), torch.cos(self.thetas[:, 2])

        R_z = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_z[:, 0, 0] = cos_a
        R_z[:, 0, 1] = -sin_a
        R_z[:, 1, 0] = sin_a
        R_z[:, 1, 1] = cos_a
        R_z[:, 2, 2] = 1

        R_y = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_y[:, 0, 0] = cos_b
        R_y[:, 0, 2] = sin_b
        R_y[:, 2, 0] = -sin_b
        R_y[:, 2, 2] = cos_b
        R_y[:, 1, 1] = 1

        R_x = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_x[:, 1, 1] = cos_g
        R_x[:, 1, 2] = -sin_g
        R_x[:, 2, 1] = sin_g
        R_x[:, 2, 2] = cos_g
        R_x[:, 0, 0] = 1

        return R_z @ R_y @ R_x
