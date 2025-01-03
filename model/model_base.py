from abc import ABC, abstractmethod


class MLMCModelBase(ABC):

    @property
    @abstractmethod
    def m(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass