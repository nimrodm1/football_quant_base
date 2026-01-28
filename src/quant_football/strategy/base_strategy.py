from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Bet:
    match_id: str
    market: str
    outcome: str
    odds: float
    prob: float
    ev: float
    stake: float

class BaseStrategy(ABC):
    def __init__(
        self, 
        value_bet_threshold: float = 0.02, 
        max_match_exposure: float = 0.05,
        **kwargs
    ):
        self.value_bet_threshold = value_bet_threshold
        self.max_match_exposure = max_match_exposure

    def calculate_bet(self, data: Dict[str, Any], bankroll: float) -> Optional[Bet]:
        """
        Calculates a single bet for a match outcome.
        """
        prob = data.get('prob', 0.0)
        odds = data.get('odds', 0.0)
        
        ev = data.get('ev', (prob * odds) - 1)
        if ev < self.value_bet_threshold:
            return None
        
        raw_stake = self.calculate_stake(data, bankroll)
        max_stake = bankroll * self.max_match_exposure
        final_stake = min(raw_stake, max_stake)
        
        if final_stake <= 0:
            return None
            
        return Bet(
            match_id=data['match_id'],
            market=data['market'],
            outcome=data['outcome'],
            odds=odds,
            prob=prob,
            ev=ev,
            stake=final_stake
        )

    def generate_bets(self, match_data_list: List[Dict[str, Any]], bankroll: float = 1000.0) -> List[Bet]:
        """
        Generates a set of bets from multiple options, picking the best per match.
        """
        best_per_match = {}
        for data in match_data_list:
            prob = data.get('prob', 0.0)
            odds = data.get('odds', 0.0)
            ev = (prob * odds) - 1
            
            mid = data['match_id']
            if mid not in best_per_match or ev > best_per_match[mid]['ev']:
                data_copy = data.copy()
                data_copy['ev'] = ev
                best_per_match[mid] = data_copy
                
        final_bets = []
        for mid, data in best_per_match.items():
            bet = self.calculate_bet(data, bankroll)
            if bet:
                final_bets.append(bet)
                
        return final_bets

    @abstractmethod
    def calculate_stake(self, data: Dict[str, Any], bankroll: float) -> float:
        pass
