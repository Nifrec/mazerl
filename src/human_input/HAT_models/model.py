import abc


class Model(metaclass=abc.ABCMeta):
    """
    Interface to be used by all implementations. The class HAT uses this interface
    to call the specified implementation as by the Strategy design pattern
    """

    @abc.abstractmethod
    def train(self):
        """train the model"""
        pass

    @abc.abstractmethod
    def forward(self):
        """perform a forward pass of the model"""
        pass
