from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from quant_football.core.config import Market, Outcomes
from quant_football.modelling.base_model import MatchPrediction

@dataclass
class Bet:
    match_id: str
    market: str
    outcome: str
    odds: float
    prob: float
    ev: float
    stake: float

@dataclass(frozen=True)
class MatchOdds:
    match_id: str
    odds: Dict[Market, Dict[Outcomes, float]] 
    metadata: Dict[str, Any] = field(default_factory=dict)

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

    def generate_bets(
        self, 
        predictions: List[MatchPrediction], 
        odds_list: List[MatchOdds], # Pass objects instead of nested dicts
        bankroll: float
    ) -> List[Bet]:
        """
        Generates a set of bets by merging predictions and odds.
        """
        # Convert odds list to a lookup for O(1) access
        odds_lookup = {o.match_id: o for o in odds_list}
        
        match_data_list = self._generate_match_data(predictions, odds_lookup)
        
        # Picking the best EV outcome per match
        best_per_match = {}
        for data in match_data_list:
            mid = data['match_id']
            ev = (data['prob'] * data['odds']) - 1
            if mid not in best_per_match or ev > best_per_match[mid]['ev']:
                candidate = data.copy()
                candidate['ev'] = ev
                best_per_match[mid] = candidate
                
        final_bets = []
        for data in best_per_match.values():
            bet = self.calculate_bet(data, bankroll)
            if bet:
                final_bets.append(bet)
                
        return final_bets

    @staticmethod
    def _generate_match_data(
        predictions: List[MatchPrediction], 
        odds_lookup: Dict[str, MatchOdds]
    ) -> List[Dict[str, Any]]:
        match_data_list = []
        for pred in predictions:
            match_odds = odds_lookup.get(pred.match_id)
            if not match_odds:
                continue
            
            for market_enum, outcomes in pred.probabilities.items():
                # Step 1: Get the market's odds sub-dictionary
                market_prices = match_odds.odds.get(market_enum, {})
                
                for outcome_enum, prob in outcomes.items():
                    # Step 2: Use the Enum object as the key for perfect alignment
                    price = market_prices.get(outcome_enum)
                    
                    if price and price > 1.0:
                        match_data_list.append({
                            "match_id": pred.match_id,
                            "market": market_enum.value,
                            # Use .value for the flattened dict/dataframe compatibility
                            "outcome": outcome_enum.value,
                            "prob": prob,
                            "odds": price,
                            **pred.metadata
                        })
                
        return match_data_list
    
    @abstractmethod
    def calculate_stake(self, data: Dict[str, Any], bankroll: float) -> float:
        pass
