from abc import ABC, abstractmethod

class MLMCModel(ABC):

    @property
    @abstractmethod
    def m(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass