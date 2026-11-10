import numpy as np

class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        shift_Z = Z-np.max(Z,axis=self.dim,keepdims = True)
        exp_Z = np.exp(shift_Z)
        # TODO: Implement forward pass
        self.A = exp_Z/np.sum(exp_Z,axis=self.dim,keepdims=True)
        return self.A
        # raise NotImplementedError

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        sum_term = np.sum(dLdA*self.A ,axis=self.dim,keepdims=True)
        dLdZ = self.A*(dLdA - sum_term)
        return dLdZ
 

    