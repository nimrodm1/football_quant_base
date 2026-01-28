import pytest
import glob
from quant_football.strategy.kelly_staking import KellyOptimalStrategy
from quant_football.strategy.flat_staking import FlatStakingStrategy

def test_stake_capping_max_exposure():
    # MATCH_003 has massive edge (0.95 * 1.5 - 1 = 0.425 EV)
    # Kelly fraction 0.25 * ((0.95*(1.5-1) - 0.05)/(1.5-1)) = 0.2125
    # 21.25% of bankroll should be capped at 5% (max_match_exposure)
    bankroll = 1000.0
    match_data = {'match_id': 'MATCH_003', 'market': 'MATCH_ODDS', 'outcome': 'HOME', 'odds': 1.5, 'prob': 0.95}
    strategy = KellyOptimalStrategy(kelly_fraction_k=0.25, max_match_exposure=0.05)
    bet = strategy.calculate_bet(match_data, bankroll)
    assert bet is not None
    assert bet.stake == pytest.approx(bankroll * 0.05)

def test_highest_ev_selection_per_match():
    # MATCH_001 HOME EV: (0.6*2.5)-1 = 0.5
    # MATCH_001 AWAY EV: (0.25*5.0)-1 = 0.25
    # Should only return one bet (HOME) for MATCH_001
    data = [
        {'match_id': 'MATCH_001', 'market': 'MATCH_ODDS', 'outcome': 'HOME', 'odds': 2.5, 'prob': 0.6},
        {'match_id': 'MATCH_001', 'market': 'MATCH_ODDS', 'outcome': 'AWAY', 'odds': 5.0, 'prob': 0.25}
    ]
    strategy = KellyOptimalStrategy()
    bets = strategy.generate_bets(data, bankroll=1000.0)
    assert len(bets) == 1
    assert bets[0].outcome == 'HOME'

def test_value_bet_threshold_filtering():
    # EV = (0.55 * 1.8) - 1 = -0.01 (Below 0.02 threshold)
    data = {'match_id': 'MATCH_004', 'market': 'MATCH_ODDS', 'outcome': 'HOME', 'odds': 1.8, 'prob': 0.55}
    strategy = KellyOptimalStrategy(value_bet_threshold=0.02)
    bet = strategy.calculate_bet(data, 1000.0)
    assert bet is None

def test_flat_staking_logic():
    bankroll = 1000.0
    match_data = {'match_id': 'MATCH_001', 'market': 'MATCH_ODDS', 'outcome': 'HOME', 'odds': 2.5, 'prob': 0.6}
    strategy = FlatStakingStrategy(flat_stake_unit=15.0)
    bet = strategy.calculate_bet(match_data, bankroll)
    assert bet.stake == 15.0

def test_kelly_zero_edge():
    # EV = (0.5 * 2.0) - 1 = 0.0. Threshold is 0.02.
    bankroll = 1000.0
    match_data = {'match_id': 'MATCH_001', 'market': 'MATCH_ODDS', 'outcome': 'HOME', 'odds': 2.0, 'prob': 0.5}
    strategy = KellyOptimalStrategy(value_bet_threshold=0.0)
    bet = strategy.calculate_bet(match_data, bankroll)
    # Kelly: (0.5*2 - 1)/(2-1) = 0
    assert bet is None

def test_kelly_odds_limit_safety():
    # Odds of 1.0 would cause division by zero in raw Kelly
    match_data = {'match_id': 'M_SAFE', 'odds': 1.0, 'prob': 0.5}
    strategy = KellyOptimalStrategy(kelly_fraction_k=0.25)
    # The strategy should handle this gracefully (return None or 0)
    bet = strategy.calculate_bet(match_data, 1000.0)
    assert bet is None

def test_kelly_overconfidence_cap():
    # Model is 100% certain, Odds are 2.0. Raw Kelly = 100% of bankroll.
    # ADDED: 'market' and 'outcome' to satisfy the Bet object requirement
    match_data = {
        'match_id': 'M_CERTAIN', 
        'market': 'MATCH_ODDS', 
        'outcome': 'HOME', 
        'odds': 2.0, 
        'prob': 1.0
    }
    strategy = KellyOptimalStrategy(max_match_exposure=0.10) # 10% Cap
    bet = strategy.calculate_bet(match_data, 1000.0)
    
    assert bet is not None
    assert bet.stake == 100.0 # Should be exactly 10% of bankroll