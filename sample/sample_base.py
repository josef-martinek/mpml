from abc import ABC, abstractmethod


class SampleBase(ABC):

    @abstractmethod
    def draw(self):
        pass