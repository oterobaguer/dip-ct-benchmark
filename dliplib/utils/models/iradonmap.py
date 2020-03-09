import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class IRadonMap(nn.Module):
    def __init__(self, ray_trafo, post_process, fully_learned, coord_mat=None):
        super(IRadonMap, self).__init__()
        self.num_detectors = ray_trafo.range.shape[-1]

        self.linear_layer = nn.Linear(in_features=self.num_detectors,
                                      out_features=self.num_detectors,
                                      bias=False)

        self.adj_ray = LearnedBackprojection(ray_trafo=ray_trafo,
                                             use_weights=fully_learned,
                                             coord_mat=coord_mat)

        self.post_process = post_process

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.adj_ray(x)
        x = self.post_process(x)
        return x


class LearnedBackprojection(nn.Module):
    def __init__(self, ray_trafo, use_weights, coord_mat=None):
        super(LearnedBackprojection, self).__init__()
        self.x_range = ray_trafo.domain.shape[0]
        self.y_range = ray_trafo.domain.shape[1]
        self.num_angles = ray_trafo.range.shape[0]
        self.num_detectors = ray_trafo.range.shape[1]

        if use_weights:
            self.weights = nn.init.ones_(Parameter(
                    torch.Tensor(1, 1, self.x_range, self.y_range,
                                 self.num_angles)))
        else:
            self.weights = 1.0

        self.coord_mat = (self.calc_coord_mat() if coord_mat is None else
                          coord_mat)

    def forward(self, sinogram):
        sinogram = sinogram.reshape((sinogram.size()[0], sinogram.size()[1],
                                     sinogram.size()[2]*sinogram.size()[3]))

        x = sinogram[:, :, self.coord_mat]
        x = torch.sum(x * self.weights, dim=-1)
        return x

    def calc_coord_mat(self):
        x_shift = int(self.x_range / 2)
        y_shift = int(self.y_range / 2)

        angle_step = np.pi / self.num_angles

        coord_matrix = np.empty((self.x_range, self.y_range,
                                self.num_angles))

        for theta in range(self.num_angles):
            for x in range(self.x_range):
                for y in range(self.y_range):
                    angle = angle_step*theta
                    coord_matrix[x, y, theta] = \
                        np.around((x - x_shift) * np.cos(angle) +
                                  (y - y_shift) * np.sin(angle))

        s_shift = np.abs(np.amin(coord_matrix))

        coord_matrix = coord_matrix + s_shift

        for theta in range(self.num_angles):
            coord_matrix[:, :, theta] = coord_matrix[:, :, theta] + \
                                        theta * self.num_detectors

        coord_matrix = coord_matrix.astype('int')

        return coord_matrix
