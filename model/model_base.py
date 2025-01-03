from abc import ABC, abstractmethod

class MLMCModel(ABC):

    @abstractmethod
    def evaluate(self):
        pass