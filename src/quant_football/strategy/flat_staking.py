from typing import Dict, Any
from .base_strategy import BaseStrategy

class FlatStakingStrategy(BaseStrategy):
    def __init__(
        self, 
        flat_stake_unit: float = 10.0,
        value_bet_threshold: float = 0.02,
        max_match_exposure: float = 0.05,
        **kwargs
    ):
        super().__init__(value_bet_threshold=value_bet_threshold, max_match_exposure=max_match_exposure)
        self.flat_stake_unit = flat_stake_unit

    def calculate_stake(self, data: Dict[str, Any], bankroll: float) -> float:
        return self.flat_stake_unit
