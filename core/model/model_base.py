from abc import ABC, abstractmethod


class ModelEvaluationBase(ABC):

    @property
    @abstractmethod
    def value(self):
        pass

    @property
    @abstractmethod
    def cost(self):
        pass


class ModelBase(ABC):

    @property
    @abstractmethod
    def m(self):
        pass

    @abstractmethod
    def get_hl(self, level):
        pass

    @abstractmethod
    def evaluate(self, level, sample) -> ModelEvaluationBase:
        pass


class MPMLModel(ModelBase):

    @abstractmethod
    def evaluate(self, level, sample, comp_tol) -> ModelEvaluationBase:
        pass