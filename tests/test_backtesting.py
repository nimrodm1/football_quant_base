import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from quant_football.core.config import BacktestConfig, Market, Outcomes
from quant_football.backtesting.backtester import Backtester
from quant_football.backtesting.pipelines import DataPipeline, ModelPipeline
from quant_football.modelling.base_model import MatchPrediction, BaseModel
from quant_football.strategy.base_strategy import BaseStrategy, Bet

@pytest.fixture
def backtest_config():
    return BacktestConfig(
        initial_bankroll=1000.0,
        training_window_months=12,
        min_training_data_points=5,
        retrain_frequency=7,
        default_odds_provider_pre_match="Avg",
        default_odds_provider_close="PS",
        value_bet_threshold=0.01
    )

@pytest.fixture
def mock_data():
    """Generates a DataFrame that mimics the output of DataLoader.load_dataset"""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    data = []
    for i, dt in enumerate(dates):
        data.append({
            "match_date": dt, # Preprocessor usually converts string to datetime
            "HomeTeam": f"Team_{i}",
            "AwayTeam": f"Opponent_{i}",
            "match_id": f"match_{i}",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
            "avg_h": 2.0, "avg_d": 3.0, "avg_a": 4.0,
            "ps_h": 2.1, "ps_d": 3.1, "ps_a": 4.1,
            "psc_h": 1.9, "psc_d": 3.2, "psc_a": 4.5
        })
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    model = MagicMock(spec=BaseModel)
    def side_effect(matches, **kwargs):
        # matches is a DataFrame with HomeTeam, AwayTeam, match_id columns
        predictions = []
        for _, row in matches.iterrows():
            predictions.append(MatchPrediction(
                match_id=row['match_id'], 
                home_team=row['HomeTeam'],
                away_team=row['AwayTeam'],
                probabilities={Market.MATCH_ODDS: {Outcomes.HOME_WIN: 0.6, Outcomes.DRAW: 0.2, Outcomes.AWAY_WIN: 0.2}}
            ))
        return predictions
    model.predict_outcome_probabilities.side_effect = side_effect
    return model

@pytest.fixture
def mock_strategy():
    strategy = MagicMock(spec=BaseStrategy)
    def gen_bets(preds, odds_list, bankroll):
        bets = []
        for p in preds:
            # Match the prediction to the odds entry
            match_odds = next((o for o in odds_list if o.match_id == p.match_id), None)
            if match_odds:
                price = match_odds.odds[Market.MATCH_ODDS][Outcomes.HOME_WIN]
                prob = p.probabilities[Market.MATCH_ODDS][Outcomes.HOME_WIN]
                bets.append(Bet(
                    match_id=p.match_id,
                    market=Market.MATCH_ODDS.value,
                    outcome=Outcomes.HOME_WIN.value,
                    odds=price,
                    prob=prob,
                    ev=(prob * price) - 1,
                    stake=10.0
                ))
        return bets
    strategy.generate_bets.side_effect = gen_bets
    return strategy

class TestBacktesting:

    def test_no_look_ahead_bias(self, backtest_config, mock_data, mock_model, mock_strategy):
        with patch.object(DataPipeline, 'prepare_data', return_value=mock_data):
            data_pipeline = DataPipeline(backtest_config)
            
            # 1. Mock the teams_mapping property
            # We use PropertyMock because 'teams_mapping' is a read-only property
            with patch.object(DataPipeline, 'teams_mapping', new_callable=PropertyMock) as mock_mapping:
                mock_mapping.return_value = {"Team_0": 0} # Return a dummy mapping
                
                model_pipeline = ModelPipeline(mock_model)
                backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)

                # 2. Update the spy signature to accept **kwargs
                def check_train_data(df, **kwargs):
                    assert (df['match_date'] < backtester.current_sim_date).all()
                
                mock_model.fit.side_effect = check_train_data
                
                # 3. Run
                backtester.run_backtest(["fake_path.csv"])

    def test_bankroll_consistency_same_day_recycling(self, backtest_config, mock_data, mock_model, mock_strategy):
        backtest_config.min_training_data_points = 1 
        
        # Setup history vs same-day matches
        historical_data = mock_data.iloc[0:5].copy()
        historical_data['match_date'] = datetime(2022, 12, 30)
        
        same_day_matches = mock_data.iloc[5:7].copy()
        same_day_matches['match_date'] = datetime(2023, 1, 1)
        
        full_test_df = pd.concat([historical_data, same_day_matches])
    
        with patch.object(DataPipeline, 'prepare_data', return_value=full_test_df):
            data_pipeline = DataPipeline(backtest_config)
            model_pipeline = ModelPipeline(mock_model)
            backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        
            bankrolls_seen = []
            original_gen_bets = mock_strategy.generate_bets.side_effect
    
            def spy_gen_bets(preds, odds, bankroll):
                bankrolls_seen.append(bankroll)
                return original_gen_bets(preds, odds, bankroll)
    
            mock_strategy.generate_bets.side_effect = spy_gen_bets
            backtester.run_backtest(["fake_path.csv"])
    
            # Should only bet on the 01/01/23 chunk
            assert len(bankrolls_seen) == 1
            assert bankrolls_seen[0] == 1000.0

    def test_retraining_frequency_logic(self, backtest_config, mock_data, mock_model, mock_strategy):
        backtest_config.retrain_frequency = 5
        with patch.object(DataPipeline, 'prepare_data', return_value=mock_data):
            data_pipeline = DataPipeline(backtest_config)
            model_pipeline = ModelPipeline(mock_model)
            backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
            backtester.run_backtest(["fake.csv"])
            
            # Logic: 30 days / 5 day frequency = 6 potential fits (depending on window)
            assert mock_model.fit.call_count >= 5

    def test_min_training_data_points_enforcement(self, backtest_config, mock_data, mock_model, mock_strategy):
        backtest_config.min_training_data_points = 100
        with patch.object(DataPipeline, 'prepare_data', return_value=mock_data):
            data_pipeline = DataPipeline(backtest_config)
            model_pipeline = ModelPipeline(mock_model)
            backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
            results = backtester.run_backtest(["fake.csv"])
            
            assert mock_model.fit.call_count == 0
            assert results['status'] == "no_bets_placed"

    def test_odds_provider_fallback_and_missing_data(self, backtest_config, mock_model, mock_strategy):
        backtest_config.default_odds_provider_pre_match = "PS"
        backtest_config.min_training_data_points = 1
    
        # Matches on Jan 2nd
        match_date = datetime(2023, 1, 2, 15, 0)
        data = [
            {
                "match_date": match_date,
                "HomeTeam": "Team_A", "AwayTeam": "Team_B",
                "match_id": "20230102_team_a_team_b",
                "FTR": "H",
                "PSH": 2.0, "PSD": 3.0, "PSA": 4.0,
                "avg_h": 2.0, "avg_d": 3.0, "avg_a": 4.0
            },
            {
                "match_date": match_date + timedelta(hours=2),
                "HomeTeam": "Team_C", "AwayTeam": "Team_D",
                "match_id": "20230102_team_c_team_d",
                "FTR": "D",
                "PSH": np.nan, "PSD": np.nan, "PSA": np.nan,
                "avg_h": 2.1, "avg_d": 3.1, "avg_a": 4.1
            }
        ]
    
        # IMPORTANT: Add a historical row so training happens BEFORE the match day
        history = {
            "match_date": datetime(2023, 1, 1, 15, 0),
            "HomeTeam": "HistH", "AwayTeam": "HistA", "match_id": "hist",
            "FTR": "H", "PSH": 2.0, "PSD": 2.0, "PSA": 2.0,
            "avg_h": 2.0, "avg_d": 2.0, "avg_a": 2.0
        }
    
        full_df = pd.DataFrame([history] + data)

        with patch.object(DataPipeline, 'prepare_data', return_value=full_df):
            data_pipeline = DataPipeline(backtest_config)
            model_pipeline = ModelPipeline(mock_model)
            backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        
            captured_odds = []
            def spy_gen_bets(preds, odds_list, bankroll):
                captured_odds.extend(odds_list)
                return [] # No bets needed for this test
            
            mock_strategy.generate_bets.side_effect = spy_gen_bets
        
            backtester.run_backtest(["fake.csv"])

            # Verification
            assert len(captured_odds) == 2
            assert captured_odds[0].odds[Market.MATCH_ODDS][Outcomes.HOME_WIN] == 2.0
            # This confirms NaN was handled as 0.0 by float(row.get(col, 0.0))
            assert captured_odds[1].odds[Market.MATCH_ODDS][Outcomes.HOME_WIN] == 0.0