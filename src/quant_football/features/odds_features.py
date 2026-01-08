import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional

from quant_football.core.config import FeatureConfig, Market
from quant_football.utils.logger import logger

#logger = logger(__name__)

class OddsFeatureEngineer:
    def __init__(self, config: FeatureConfig):
        self.config = config

    def _get_market_odds_columns(self, df: pd.DataFrame, market_type: Market, period: str) -> Dict[str, Dict[str, str]]:
        """
        Helper method to identify odds columns for a given market type and period (pre_match/closing)
        using config.ODDS_COL_PATTERNS.
        Returns a dictionary mapping bookmaker prefixes to a dictionary of odds_type (home, draw, away, over, under)
        and their corresponding column names present in the DataFrame.
        Example: {"B365": {"home": "B365H", "draw": "B365D", "away": "B365A"}}
        """
        market_patterns = self.config.ODDS_COL_PATTERNS.get(market_type, {}).get(period, {})
        
        found_odds_columns: Dict[str, Dict[str, str]] = {}

        for bookmaker_prefix, pattern in market_patterns.items():
            bookmaker_odds = {}
            if market_type == Market.MATCH_ODDS:
                # For match odds, we expect H, D, A
                for suffix, odds_type in [("H", "home"), ("D", "draw"), ("A", "away")]:
                    # Construct specific pattern for H, D, A
                    specific_pattern = pattern.replace("[HDA]", suffix)
                    matching_cols = [col for col in df.columns if re.match(specific_pattern, col)]
                    if matching_cols:
                        # Assuming only one column matches per specific pattern
                        bookmaker_odds[odds_type] = matching_cols[0]
            elif market_type == Market.OVER_UNDER_2_5:
                # For over/under 2.5, we expect >2.5, <2.5
                for suffix, odds_type in [(">2.5", "over"), ("<2.5", "under")]:
                    specific_pattern = pattern.replace("[<>]2.5", suffix) # Replace generic part
                    matching_cols = [col for col in df.columns if re.match(specific_pattern, col)]
                    if matching_cols:
                        bookmaker_odds[odds_type] = matching_cols[0]
            elif market_type == Market.ASIAN_HANDICAP:
                # For Asian Handicap, we expect AHh, AHH, AHA or CAHh, CAHH, CAHA etc.
                # The pattern itself contains the specific suffix, so we directly match.
                if bookmaker_prefix == "AHh" or bookmaker_prefix == "AHCh": # This is a single column
                    matching_cols = [col for col in df.columns if re.match(pattern, col)]
                    if matching_cols:
                        bookmaker_odds["handicap_line"] = matching_cols[0]
                else: # For bookmaker specific AH odds
                    for suffix, odds_type in [("AHH", "home"), ("AHA", "away")]:
                        specific_pattern = pattern.replace("[HA]", suffix[-1]) # Replace [HA] with H or A
                        matching_cols = [col for col in df.columns if re.match(specific_pattern, col)]
                        if matching_cols:
                            bookmaker_odds[odds_type] = matching_cols[0]

            if bookmaker_odds:
                found_odds_columns[bookmaker_prefix] = bookmaker_odds
        
        return found_odds_columns

    def calculate_implied_probabilities(self, df: pd.DataFrame, bookmaker_prefix: str) -> pd.DataFrame:
        """
        Calculates normalized implied probabilities from match odds.
        Includes a sanity check on the overround to handle corrupt data (e.g. near-zero odds).
        """
        df_copy = df.copy()
        
        pre_match_odds_map = self._get_market_odds_columns(df_copy, Market.MATCH_ODDS, "pre_match")
        bookmaker_odds = pre_match_odds_map.get(bookmaker_prefix, {})

        home_col = bookmaker_odds.get("home")
        draw_col = bookmaker_odds.get("draw")
        away_col = bookmaker_odds.get("away")

        if not all([home_col, draw_col, away_col]):
            logger.warning(f"Missing columns for {bookmaker_prefix}. Skipping.")
            for suffix in ['home', 'draw', 'away']:
                df_copy[f'implied_prob_{suffix}_{bookmaker_prefix}'] = np.nan
            return df_copy

        # 1. Calculate raw implied probabilities (1/decimal_odds)
        # Use a small epsilon or replace(0) to avoid division by zero
        prob_home = 1 / df_copy[home_col].replace(0, np.nan)
        prob_draw = 1 / df_copy[draw_col].replace(0, np.nan)
        prob_away = 1 / df_copy[away_col].replace(0, np.nan)

        # 2. Calculate the Overround (Sum of raw probabilities)
        sum_probs = prob_home + prob_draw + prob_away

        # 3. Sanity Check: Identify "garbage" data
        # Thresholds: 
        # - Lower: 0.95 (allows for slight arbs/promos)
        # - Upper: 2.00 (anything requiring a 100% margin is likely a data error)
        is_sane = (sum_probs >= 0.95) & (sum_probs <= 2.0)
        
        # 4. Normalize and apply mask
        # We only divide by sum_probs where the data is sane and non-zero
        valid_mask = is_sane & (sum_probs > 0)
        
        df_copy[f'implied_prob_home_{bookmaker_prefix}'] = np.where(valid_mask, prob_home / sum_probs, np.nan)
        df_copy[f'implied_prob_draw_{bookmaker_prefix}'] = np.where(valid_mask, prob_draw / sum_probs, np.nan)
        df_copy[f'implied_prob_away_{bookmaker_prefix}'] = np.where(valid_mask, prob_away / sum_probs, np.nan)
        
        logger.info(f"Calculated implied probabilities for {bookmaker_prefix} (Sane rows: {is_sane.sum()}/{len(df)})")
        return df_copy

    def calculate_implied_probabilities_over_under(self, df: pd.DataFrame, bookmaker_prefix: str) -> pd.DataFrame:
        """
        Calculates implied probabilities (implied_prob_over_2_5, _under_2_5) from pre-match odds.
        Includes a sanity check on the overround to handle corrupt data.
        """
        df_copy = df.copy()
        
        pre_match_odds_map = self._get_market_odds_columns(df_copy, Market.OVER_UNDER_2_5, "pre_match")
        bookmaker_odds = pre_match_odds_map.get(bookmaker_prefix, {})

        over_col = bookmaker_odds.get("over")
        under_col = bookmaker_odds.get("under")

        if not all([over_col, under_col]):
            logger.warning(f"Missing O/U columns for {bookmaker_prefix}. Skipping.")
            df_copy[f'implied_prob_over_2_5_{bookmaker_prefix}'] = np.nan
            df_copy[f'implied_prob_under_2_5_{bookmaker_prefix}'] = np.nan
            return df_copy

        # 1. Calculate raw implied probabilities
        prob_over = 1 / df_copy[over_col].replace(0, np.nan)
        prob_under = 1 / df_copy[under_col].replace(0, np.nan)

        # 2. Calculate the Overround (Sum of probabilities)
        sum_probs = prob_over + prob_under

        # 3. Sanity Check: Identify "garbage" data
        # Thresholds: 0.95 (allows for promos/arbs) to 2.0 (filters corrupt data)
        is_sane = (sum_probs >= 0.95) & (sum_probs <= 2.0)
        
        # 4. Normalize and apply mask
        valid_mask = is_sane & (sum_probs > 0)
        
        df_copy[f'implied_prob_over_2_5_{bookmaker_prefix}'] = np.where(
            valid_mask, prob_over / sum_probs, np.nan
        )
        df_copy[f'implied_prob_under_2_5_{bookmaker_prefix}'] = np.where(
            valid_mask, prob_under / sum_probs, np.nan
        )

        logger.info(f"Calculated O/U implied probabilities for {bookmaker_prefix} (Sane rows: {is_sane.sum()}/{len(df)})")
        return df_copy

    def get_best_pre_match_odds(self, df: pd.DataFrame, market_type: Market) -> pd.DataFrame:
        """
        Finds the best available pre-match odds across all bookmakers for the specified market_type.
        Preserves original columns.
        """
        df_copy = df.copy()
        pre_match_odds_map = self._get_market_odds_columns(df_copy, market_type, "pre_match")

        if not pre_match_odds_map:
            logger.warning(f"No pre-match odds found for market type {market_type}. Skipping best odds calculation.")
            if market_type == Market.MATCH_ODDS:
                df_copy['best_odds_home'] = np.nan
                df_copy['best_odds_draw'] = np.nan
                df_copy['best_odds_away'] = np.nan
            elif market_type == Market.OVER_UNDER_2_5:
                df_copy['best_odds_over_2_5'] = np.nan
                df_copy['best_odds_under_2_5'] = np.nan
            return df_copy
        
        if market_type == Market.MATCH_ODDS:
            home_odds_cols = [odds_data["home"] for odds_data in pre_match_odds_map.values() if "home" in odds_data]
            draw_odds_cols = [odds_data["draw"] for odds_data in pre_match_odds_map.values() if "draw" in odds_data]
            away_odds_cols = [odds_data["away"] for odds_data in pre_match_odds_map.values() if "away" in odds_data]

            if home_odds_cols: df_copy['best_odds_home'] = df_copy[home_odds_cols].max(axis=1)
            else: df_copy['best_odds_home'] = np.nan
            if draw_odds_cols: df_copy['best_odds_draw'] = df_copy[draw_odds_cols].max(axis=1)
            else: df_copy['best_odds_draw'] = np.nan
            if away_odds_cols: df_copy['best_odds_away'] = df_copy[away_odds_cols].max(axis=1)
            else: df_copy['best_odds_away'] = np.nan
            logger.info(f"Calculated best pre-match match odds.")

        elif market_type == Market.OVER_UNDER_2_5:
            over_odds_cols = [odds_data["over"] for odds_data in pre_match_odds_map.values() if "over" in odds_data]
            under_odds_cols = [odds_data["under"] for odds_data in pre_match_odds_map.values() if "under" in odds_data]
            
            if over_odds_cols: df_copy['best_odds_over_2_5'] = df_copy[over_odds_cols].max(axis=1)
            else: df_copy['best_odds_over_2_5'] = np.nan
            if under_odds_cols: df_copy['best_odds_under_2_5'] = df_copy[under_odds_cols].max(axis=1)
            else: df_copy['best_odds_under_2_5'] = np.nan
            logger.info(f"Calculated best pre-match Over/Under 2.5 odds.")

        return df_copy

    def get_closing_odds(self, df: pd.DataFrame, market_type: Market, provider_prefix: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves closing odds with fallback logic. Preserves original columns.
        """
        df_copy = df.copy()
        closing_odds_map = self._get_market_odds_columns(df_copy, market_type, "closing")

        if market_type == Market.MATCH_ODDS:
            preferred_home_col, preferred_draw_col, preferred_away_col = None, None, None
            max_home_col, max_draw_col, max_away_col = None, None, None

            # Identify Max odds columns
            max_odds_data = closing_odds_map.get("Max", {})
            max_home_col = max_odds_data.get("home")
            max_draw_col = max_odds_data.get("draw")
            max_away_col = max_odds_data.get("away")

            if provider_prefix:
                preferred_odds_data = closing_odds_map.get(provider_prefix, {})
                preferred_home_col = preferred_odds_data.get("home")
                preferred_draw_col = preferred_odds_data.get("draw")
                preferred_away_col = preferred_odds_data.get("away")

                if not all([preferred_home_col, preferred_draw_col, preferred_away_col]):
                    logger.warning(f"Preferred provider ({provider_prefix}) closing odds columns not fully found for Match Odds. Falling back to Max odds.")
                    df_copy['closing_odds_home'] = df_copy[max_home_col] if max_home_col else np.nan
                    df_copy['closing_odds_draw'] = df_copy[max_draw_col] if max_draw_col else np.nan
                    df_copy['closing_odds_away'] = df_copy[max_away_col] if max_away_col else np.nan
                else:
                    # Apply fallback for each odds type individually if preferred is NaN
                    df_copy['closing_odds_home'] = df_copy[preferred_home_col].fillna(df_copy[max_home_col]) if max_home_col else df_copy[preferred_home_col]
                    df_copy['closing_odds_draw'] = df_copy[preferred_draw_col].fillna(df_copy[max_draw_col]) if max_draw_col else df_copy[preferred_draw_col]
                    df_copy['closing_odds_away'] = df_copy[preferred_away_col].fillna(df_copy[max_away_col]) if max_away_col else df_copy[preferred_away_col]
                logger.info(f"Calculated closing match odds using preferred provider {provider_prefix} with Max fallback.")

            else: # No preferred provider, use Max odds
                if not all([max_home_col, max_draw_col, max_away_col]):
                    logger.warning("Max closing odds columns not fully found for Match Odds. Closing odds will be NaN.")
                    df_copy['closing_odds_home'] = np.nan
                    df_copy['closing_odds_draw'] = np.nan
                    df_copy['closing_odds_away'] = np.nan
                else:
                    df_copy['closing_odds_home'] = df_copy[max_home_col]
                    df_copy['closing_odds_draw'] = df_copy[max_draw_col]
                    df_copy['closing_odds_away'] = df_copy[max_away_col]
                logger.info(f"Calculated closing match odds using Max provider.")
        
        # For other market types, similar logic would apply if closing odds are needed
        # For now, only Match Odds closing odds are specified in the plan.
        return df_copy