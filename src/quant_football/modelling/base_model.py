from abc import ABC, abstractmethod
from quant_football.core.config import Market 
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd

@dataclass(frozen=True)
class MatchPrediction:
    match_id: str
    home_team: str
    away_team: str
    # Market name -> Outcome -> Probability
    # e.g., {"1X2": {"home_win": 0.45, "draw": 0.25, ...}, "OU2.5": {"over": 0.55, ...}}
    probabilities: Dict[Market, Dict[str, float]] 
    metadata: Dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> List[Dict[str, Any]]:
        rows = []
        for market_enum, outcomes in self.probabilities.items():
            for outcome, prob in outcomes.items():
                rows.append({
                    "match_id": self.match_id,
                    "market": market_enum.value, # Use the string value for the DF
                    "outcome": outcome,
                    "prob": prob,
                    **self.metadata
                })
        return rows

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
