from contextlib import closing
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any, Optional

from pyparsing import col
from quant_football.core.config import BacktestConfig, Market, Outcomes
from quant_football.modelling.base_model import MatchPrediction
from quant_football.strategy.base_strategy import BaseStrategy, MatchOdds, Bet
from quant_football.backtesting.pipelines import DataPipeline, ModelPipeline
from quant_football.utils.logger import logger

class Backtester:
    def __init__(
        self, 
        config: BacktestConfig,
        data_pipeline: DataPipeline,
        model_pipeline: ModelPipeline,
        strategy: BaseStrategy
    ):
        self.config = config
        self.data_pipeline = data_pipeline
        self.model_pipeline = model_pipeline
        self.strategy = strategy
        self.bankroll = config.initial_bankroll
        self.history: List[Dict[str, Any]] = []
        self.bankroll_history: List[Dict[str, Any]] = []
        self.current_sim_date: Optional[pd.Timestamp] = None

    def run_backtest(self, df: pd.DataFrame, eval_only: bool = False):
        """
        Main loop for walk-forward backtesting.
        """
        processed_df = self.data_pipeline.prepare_data(df)
        
        if processed_df.empty:
            logger.warning("No data to backtest after preprocessing.")
            return self.evaluate_performance()

        # Determine simulation timeframe
        start_date = processed_df['match_date'].min().floor('D')
        end_date = processed_df['match_date'].max().floor('D')
        
        self.current_sim_date = start_date
        last_retrain_date = None

        while self.current_sim_date <= end_date:
            # 1. Retrain Logic
            if self._should_retrain(self.current_sim_date, last_retrain_date):
                train_data = self._get_training_window(self.current_sim_date, processed_df)
                if len(train_data) >= self.config.min_training_data_points:
                    logger.info(f"Retraining model on {self.current_sim_date} with {len(train_data)} samples.")
                    global_mapping = self.data_pipeline.teams_mapping
                    self.model_pipeline.train(train_data, teams_mapping=global_mapping)
                    logger.info("Model retrained successfully.")
                    last_retrain_date = self.current_sim_date

            # 2. Daily Matches
            daily_matches = processed_df[
                (processed_df['match_date'] >= self.current_sim_date) & 
                (processed_df['match_date'] < self.current_sim_date + timedelta(days=1))
            ]

            if not daily_matches.empty and last_retrain_date is not None:
                self._process_daily_matches(daily_matches, self.current_sim_date, eval_only)

            # 3. Snapshot Bankroll
            self.bankroll_history.append({
                "date": self.current_sim_date,
                "bankroll": self.bankroll
            })
            logger.info(f"Date: {self.current_sim_date.date()}, Bankroll: {self.bankroll:.2f}, Bets Placed: {len(daily_matches)}")

            self.current_sim_date += timedelta(days=1)

        return self.evaluate_performance()

    def _should_retrain(self, current_date, last_retrain_date) -> bool:
        if last_retrain_date is None:
            return True
        days_since = (current_date - last_retrain_date).days
        return days_since >= self.config.retrain_frequency

    def _get_training_window(self, current_date, df) -> pd.DataFrame:
        cutoff = current_date - timedelta(days=self.config.training_window_months * 30)
        return df[(df['match_date'] < current_date) & (df['match_date'] >= cutoff)]

    def _process_daily_matches(self, matches: pd.DataFrame, current_date: pd.Timestamp, eval_only: bool):
        # Get predictions for all matches at once
        predictions = self.model_pipeline.predict(matches)
        
        # Create mappings for efficient lookup
        pred_map = {p.match_id: p for p in predictions}
        row_map = {row['match_id']: row for _, row in matches.iterrows()}
        
        odds_list = []
        match_results = {}
        match_closing_odds = {}

        for match_id, row in row_map.items():
            # Pre-match odds
            pre_match_odds_dict = self._extract_odds(row, closing=False)
            odds_list.append(MatchOdds(match_id=match_id, odds=pre_match_odds_dict))
            
            # Closing odds
            closing_odds_dict = self._extract_odds(row, closing=True)
            match_closing_odds[match_id] = closing_odds_dict
            
            # Real Result
            match_results[match_id] = row['FTR']

        if eval_only:
            # Eval Only Mode: Record all outcomes
            for pred in predictions:
                result = match_results.get(pred.match_id)
                row = row_map.get(pred.match_id)
                if row is None:
                    continue
                    
                mapping = {"H": Outcomes.HOME_WIN, "D": Outcomes.DRAW, "A": Outcomes.AWAY_WIN}
                actual_outcome = mapping.get(result)
                
                for market, outcomes in pred.probabilities.items():
                    if market == Market.MATCH_ODDS:
                        for outcome, prob in outcomes.items():
                            self.history.append({
                                "match_id": pred.match_id,
                                "date": current_date,
                                "home_team": row['HomeTeam'], 
                                "away_team": row['AwayTeam'],  
                                "league": row.get('Div', 'Unknown'),
                                "market": market.value,
                                "outcome": outcome.value,
                                "prob": prob,
                                "is_win": (outcome == actual_outcome),
                                "stake": 0.0,
                                "pnl": 0.0,
                                "odds": 0.0,
                                "closing_odds": 0.0
                            })
        else:
            # Full Backtest Mode
            bets = self.strategy.generate_bets(predictions, odds_list, self.bankroll)

            for bet in bets:
                row = row_map.get(bet.match_id)
                if row is None:
                    continue
                    
                result = match_results.get(bet.match_id)
                is_win = self._check_win(bet, result)
                
                pnl = (bet.stake * (bet.odds - 1)) if is_win else -bet.stake
                self.bankroll += pnl
                
                closing_prices = match_closing_odds.get(bet.match_id, {})
                market_enum = Market(bet.market)
                outcome_enum = Outcomes(bet.outcome)
                closing_price = closing_prices.get(market_enum, {}).get(outcome_enum, np.nan)
                
                self.history.append({
                    "match_id": bet.match_id,
                    "date": current_date,
                    "home_team": row['HomeTeam'], 
                    "away_team": row['AwayTeam'],  
                    "league": row.get('Div', 'Unknown'),
                    "market": bet.market,
                    "outcome": bet.outcome,
                    "odds": bet.odds,
                    "closing_odds": closing_price,
                    "prob": bet.prob,
                    "stake": bet.stake,
                    "pnl": pnl,
                    "is_win": is_win,
                    "actual_result": result
                })

    def _extract_odds(self, row: pd.Series, closing: bool = False) -> Dict[Market, Dict[Outcomes, float]]:
        if closing:
            provider = self.config.default_odds_provider_close
            h_col, d_col, a_col = f"{provider}CH", f"{provider}CD", f"{provider}CA"
        else:
            provider = self.config.default_odds_provider_pre_match
            h_col, d_col, a_col = f"{provider}H", f"{provider}D", f"{provider}A"
    
        # helper to handle both missing columns and NaN values
        def get_val(col):
            val = row.get(col, 0.0)
            return float(val) if pd.notna(val) else 0.0

        return {
            Market.MATCH_ODDS: {
                Outcomes.HOME_WIN: get_val(h_col),
                Outcomes.DRAW: get_val(d_col),
                Outcomes.AWAY_WIN: get_val(a_col)
            }
        }

    def _check_win(self, bet: Bet, ftr: str) -> bool:
        mapping = {
            "H": Outcomes.HOME_WIN.value,
            "D": Outcomes.DRAW.value,
            "A": Outcomes.AWAY_WIN.value
        }
        return bet.outcome == mapping.get(ftr)

    def evaluate_performance(self) -> Dict[str, Any]:
        if not self.history:
            return {"status": "no_bets_placed", "bankroll": self.bankroll}

        results_df = pd.DataFrame(self.history)
        
        total_staked = results_df['stake'].sum()
        total_pnl = results_df['pnl'].sum()
        
        metrics = {
            'status': "success",
            "roi": total_pnl / total_staked if total_staked > 0 else 0,
            "pnl": total_pnl,
            "final_bankroll": self.bankroll,
            "total_staked": total_staked,
            "win_rate": results_df['is_win'].mean(),
            "count": len(results_df)
        }
        
        y_true = results_df['is_win'].astype(int)
        y_prob = results_df['prob']
        
        metrics['brier_score'] = np.mean((y_prob - y_true)**2)
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        metrics['log_loss'] = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
        
        valid_clv = results_df[results_df['closing_odds'] > 1.0]
        if not valid_clv.empty:
            metrics['clv_score'] = ((valid_clv['odds'] / valid_clv['closing_odds']) - 1).mean()
        else:
            metrics['clv_score'] = np.nan

        return metrics
