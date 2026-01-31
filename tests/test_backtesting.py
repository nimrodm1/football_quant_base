import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

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
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    data = []
    for i, dt in enumerate(dates):
        data.append({
            "Div": "E0",
            "Date": dt.strftime("%d/%m/%y"),
            "Time": "15:00",
            "HomeTeam": f"Team_{i}",
            "AwayTeam": f"Opponent_{i}",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
            "AvgH": 2.0, "AvgD": 3.0, "AvgA": 4.0,
            "PSH": 2.1, "PSD": 3.1, "PSA": 4.1,
            "PSCH": 1.9, "PSCD": 3.2, "PSCA": 4.5
        })
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    model = MagicMock(spec=BaseModel)
    def side_effect(home, away, **kwargs):
        return [MatchPrediction(
            match_id="dummy", 
            home_team=home,
            away_team=away,
            probabilities={Market.MATCH_ODDS: {Outcomes.HOME_WIN: 0.6, Outcomes.DRAW: 0.2, Outcomes.AWAY_WIN: 0.2}}
        )]
    model.predict_outcome_probabilities.side_effect = side_effect
    return model

@pytest.fixture
def mock_strategy():
    strategy = MagicMock(spec=BaseStrategy)
    def gen_bets(preds, odds_list, bankroll):
        bets = []
        for p in preds:
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
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        def check_train_data(df):
            assert (df['match_date'] < backtester.current_sim_date).all()
        mock_model.fit.side_effect = check_train_data
        backtester.run_backtest(mock_data)

    def test_bankroll_consistency_same_day_recycling(self, backtest_config, mock_data, mock_model, mock_strategy):
        # 1. Lower the gate
        backtest_config.min_training_data_points = 1 
    
        # 2. Setup History (December)
        historical_data = mock_data.iloc[0:5].copy()
        historical_data['Date'] = "30/12/22" 
    
        # 3. Setup Two Matches for the SAME day (January 1st)
        same_day_matches = mock_data.iloc[5:7].copy()
        same_day_matches['Date'] = "01/01/23"
    
        # Combine them: Total 7 rows
        full_test_df = pd.concat([historical_data, same_day_matches])
    
        # Initialise components
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
    
        bankrolls_seen = []
        original_gen_bets = mock_strategy.generate_bets.side_effect
    
        def spy_gen_bets(preds, odds, bankroll):
            bankrolls_seen.append(bankroll)
            return original_gen_bets(preds, odds, bankroll)
    
        mock_strategy.generate_bets.side_effect = spy_gen_bets
    
        # 4. RUN with the FULL dataframe
        backtester.run_backtest(full_test_df)
    
        # 5. VERIFY
        # We expect 1 call to generate_bets for 01/01/23 containing BOTH matches.
        # (The historical dates won't trigger bets because they are used for training)
        assert len(bankrolls_seen) == 1
        assert bankrolls_seen[0] == 1000.0

    def test_retraining_frequency_logic(self, backtest_config, mock_data, mock_model, mock_strategy):
        backtest_config.retrain_frequency = 5
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        backtester.run_backtest(mock_data)
        assert mock_model.fit.call_count == 5

    def test_min_training_data_points_enforcement(self, backtest_config, mock_data, mock_model, mock_strategy):
        backtest_config.min_training_data_points = 100
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        results = backtester.run_backtest(mock_data)
        assert mock_model.fit.call_count == 0
        assert results['status'] == "no_bets_placed"

    def test_correct_odds_provider_usage_pre_vs_close(self, backtest_config, mock_data, mock_model, mock_strategy):
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        backtester.run_backtest(mock_data)
        for entry in backtester.history:
            assert entry['odds'] == 2.0
            assert entry['closing_odds'] == 1.9

    def test_eval_only_mode_calculates_brier_without_staking(self, backtest_config, mock_data, mock_model, mock_strategy):
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        metrics = backtester.run_backtest(mock_data, eval_only=True)
        assert 'brier_score' in metrics
        assert metrics['total_staked'] == 0
        assert backtester.bankroll == 1000.0

    def test_eval_metrics_calculation(self, backtest_config, mock_data, mock_model, mock_strategy):
        data_pipeline = DataPipeline(backtest_config)
        model_pipeline = ModelPipeline(mock_model)
        backtester = Backtester(backtest_config, data_pipeline, model_pipeline, mock_strategy)
        metrics = backtester.run_backtest(mock_data)
        assert 'roi' in metrics
        assert 'pnl' in metrics
        assert 'brier_score' in metrics
        assert 'log_loss' in metrics
        assert 'clv_score' in metrics
        assert metrics['count'] > 0
