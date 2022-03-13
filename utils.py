import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
import numpy as np
import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

class ClusterAssignment(nn.Module):

    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = torch.Tensor(3, 25088)
        #cluster_centers: Optional[torch.Tensor] = None
    ) -> torch.tensor:

        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            x = nn.init.xavier_uniform_(initial_cluster_centers)
            print("xavier cluster centers")
            print(x)
            print(x.shape)
        else:
            initial_cluster_centers = cluster_centers
            print("cluster centers in the cluster assignment")
            print(initial_cluster_centers)
        self.cluster_centers = Parameter(initial_cluster_centers) # I Need this point to execute next run

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feature = torch.flatten(batch, 1)
        norm_squared = torch.sum((feature.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True) # I could see with seeing what this value is




def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

