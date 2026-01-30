import pytest
from typing import Dict, Any
from quant_football.strategy import BaseStrategy, KellyOptimalStrategy, FlatStakingStrategy, MatchOdds
from quant_football.modelling import MatchPrediction
from quant_football.core.config import Market, Outcomes

# --- UNIT TESTS (Testing Math & Capping) ---

def test_stake_capping_max_exposure():
    bankroll = 1000.0
    # Dictionary format is fine for internal calculate_bet testing
    match_data = {
        'match_id': 'MATCH_003', 
        'market': 'MATCH_ODDS', 
        'outcome': 'home_win', 
        'odds': 1.5, 
        'prob': 0.95
    }
    strategy = KellyOptimalStrategy(kelly_fraction_k=0.25, max_match_exposure=0.05)
    bet = strategy.calculate_bet(match_data, bankroll)
    
    assert bet is not None
    # 21.25% raw Kelly should be capped at 5% of 1000
    assert bet.stake == pytest.approx(50.0)

def test_value_bet_threshold_filtering():
    # EV = (0.55 * 1.8) - 1 = -0.01 (Below 0.02 threshold)
    data = {'match_id': 'M4', 'market': 'MATCH_ODDS', 'outcome': 'home_win', 'odds': 1.8, 'prob': 0.55}
    strategy = KellyOptimalStrategy(value_bet_threshold=0.02)
    bet = strategy.calculate_bet(data, 1000.0)
    assert bet is None

def test_flat_staking_logic():
    bankroll = 1000.0
    match_data = {'match_id': 'M1', 'market': 'MATCH_ODDS', 'outcome': 'home_win', 'odds': 2.5, 'prob': 0.6}
    strategy = FlatStakingStrategy(flat_stake_unit=15.0)
    bet = strategy.calculate_bet(match_data, bankroll)
    assert bet.stake == 15.0

def test_kelly_zero_edge():
    bankroll = 1000.0
    match_data = {'match_id': 'M1', 'market': 'MATCH_ODDS', 'outcome': 'home_win', 'odds': 2.0, 'prob': 0.5}
    strategy = KellyOptimalStrategy(value_bet_threshold=0.0)
    bet = strategy.calculate_bet(match_data, bankroll)
    assert bet is None

def test_kelly_odds_limit_safety():
    # Test that odds of 1.0 or less are handled gracefully
    match_data = {'match_id': 'M_SAFE', 'market': 'MATCH_ODDS', 'outcome': 'home_win', 'odds': 1.0, 'prob': 0.5}
    strategy = KellyOptimalStrategy(kelly_fraction_k=0.25)
    bet = strategy.calculate_bet(match_data, 1000.0)
    assert bet is None

def test_kelly_overconfidence_cap():
    match_data = {
        'match_id': 'M_CERTAIN', 
        'market': 'MATCH_ODDS', 
        'outcome': 'home_win', 
        'odds': 2.0, 
        'prob': 1.0
    }
    strategy = KellyOptimalStrategy(max_match_exposure=0.10) 
    bet = strategy.calculate_bet(match_data, 1000.0)
    assert bet.stake == 100.0 

# --- INTEGRATION TESTS (Testing the Handshake & Selection) ---

def test_highest_ev_selection_per_match():
    match_id = 'MATCH_001'
    bankroll = 1000.0

    predictions = [
        MatchPrediction(
            match_id=match_id,
            home_team="Team A",
            away_team="Team B",
            probabilities={
                Market.MATCH_ODDS: {
                    Outcomes.HOME_WIN: 0.6,   # EV: 0.5
                    Outcomes.AWAY_WIN: 0.25  # EV: 0.25
                }
            }
        )
    ]

    odds_data = [
        MatchOdds(
            match_id=match_id,
            odds={
                Market.MATCH_ODDS: {
                    Outcomes.HOME_WIN: 2.5,
                    Outcomes.AWAY_WIN: 5.0
                }
            }
        )
    ]

    strategy = KellyOptimalStrategy(value_bet_threshold=0.01)
    bets = strategy.generate_bets(predictions, odds_data, bankroll)

    assert len(bets) == 1
    # Check that it picked HOME_WIN (higher EV) over AWAY_WIN
    assert bets[0].outcome == Outcomes.HOME_WIN.value

def test_strategy_handshake_alignment():
    """Validates that nested enums are flattened correctly into strings."""
    match_id = "LIV_ARS"
    prediction = MatchPrediction(
        match_id=match_id,
        home_team="Liverpool",
        away_team="Arsenal",
        probabilities={Market.OVER_UNDER_2_5: {Outcomes.OVER_25: 0.70}}
    )
    odds = MatchOdds(
        match_id=match_id,
        odds={Market.OVER_UNDER_2_5: {Outcomes.OVER_25: 2.0}}
    )
    
    class MockStrategy(BaseStrategy):
        def calculate_stake(self, d, b): return 10.0

    strategy = MockStrategy(value_bet_threshold=0.01)
    bets = strategy.generate_bets([prediction], [odds], 1000.0)
    
    assert bets[0].market == Market.OVER_UNDER_2_5.value
    assert bets[0].outcome == Outcomes.OVER_25.value

def test_strategy_filters_negative_ev_integration():
    match_id = "bad_bet"
    prediction = MatchPrediction(
        match_id=match_id,
        home_team="A", away_team="B",
        probabilities={Market.MATCH_ODDS: {Outcomes.HOME_WIN: 0.40}}
    )
    odds = MatchOdds(
        match_id=match_id,
        odds={Market.MATCH_ODDS: {Outcomes.HOME_WIN: 2.0}} # EV = 0.0
    )
    
    strategy = KellyOptimalStrategy(value_bet_threshold=0.02)
    bets = strategy.generate_bets([prediction], [odds], 1000.0)
    
    assert len(bets) == 0