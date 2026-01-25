from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class BaseModel(ABC):
    """
    Abstract base class for all predictive models in quant_football.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.trace = None

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Train the model using the provided data.
        """
        pass

    @abstractmethod
    def predict_outcome_probabilities(self, home_team: str, away_team: str, **kwargs) -> Dict[str, float]:
        """
        Predict outcome probabilities (Home, Draw, Away) for a given match.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Persist the model to a file.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the model from a file.
        """
        pass
