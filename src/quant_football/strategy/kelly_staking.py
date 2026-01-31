from typing import Dict, Any
from .base_strategy import BaseStrategy

class KellyOptimalStrategy(BaseStrategy):
    def __init__(
        self, 
        kelly_fraction_k: float = 0.25,
        value_bet_threshold: float = 0.02,
        max_match_exposure: float = 0.05,
        **kwargs
    ):
        super().__init__(value_bet_threshold=value_bet_threshold, max_match_exposure=max_match_exposure)
        self.kelly_fraction_k = kelly_fraction_k

    def calculate_stake(self, data: Dict[str, Any], bankroll: float) -> float:
        """
        Kelly Criterion: f* = (p(b-1) - q) / (b-1) = (p*b - 1) / (b-1)
        where b is decimal odds.
        """
        p = data['prob']
        b = data['odds']
        
        if b <= 1:
            return 0.0
            
        kelly_f = (p * b - 1) / (b - 1)
        
        if kelly_f <= 0:
            return 0.0
            
        stake = bankroll * kelly_f * self.kelly_fraction_k
        return stake
