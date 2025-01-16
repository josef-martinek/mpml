from abc import ABC, abstractmethod


class LinsorverBase(ABC):

    @abstractmethod
    def solve_system(self):
        pass