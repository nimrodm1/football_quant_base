import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
import re

from quant_football.core.config import DataConfig, Market
from quant_football.utils.logger import logger

class Preprocessor:
    def __init__(self, config: DataConfig):
        self.config = config

    def clean_and_standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        # Essential order: Rename -> Standardise Names -> Parse Dates -> Enforce Schema -> Validate
        df = self.handle_inconsistent_columns(df)
        df.rename(columns=self.config.RAW_COL_MAP, inplace=True)
        
        df = self.standardise_team_names(df)
        df = self._parse_dates(df)
        df = self._enforce_schema(df)
        df = self._validate_critical_data(df)
    
        if 'match_date' in df.columns:
            df.sort_values(by='match_date', inplace=True, ignore_index=True)
        # At the very end of clean_and_standardise:
        cols_to_drop = ['Date', 'Time']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        return df

    def standardise_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the mapping from config to ensure team name consistency.
        This is the best place to add new mappings as you find them.
        """
        for col in ['HomeTeam', 'AwayTeam']:
            if col in df.columns:
            # We use the mapping from your config
            # It's wise to keep this dictionary growing as you find new variations
                df[col] = df[col].replace(self.config.TEAM_NAME_MAPPING)
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses Date and Time columns into a single match_date with explicit formatting.
        """
        if 'Date' not in df.columns:
            return df

        # 1. Prepare the time string
        times = df['Time'].fillna('12:00').astype(str) if 'Time' in df.columns else '12:00'
        dt_strings = df['Date'].astype(str) + ' ' + times

        # 2. Use an explicit format to stop the UserWarning
        # Format explanation: %d/%m/%y is 01/01/21, %H:%M is 15:30
        # We try the most common format first (DD/MM/YY HH:MM)
        df['match_date'] = pd.to_datetime(
            dt_strings, 
            format='%d/%m/%y %H:%M', 
            errors='coerce'
        )

        # 3. Fallback for 4-digit years (DD/MM/YYYY) if any NaT (Not a Time) remain
        if df['match_date'].isna().any():
            mask = df['match_date'].isna()
            df.loc[mask, 'match_date'] = pd.to_datetime(
                dt_strings[mask], 
                format='%d/%m/%Y %H:%M', 
                errors='coerce'
            )
        
        return df

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the DataFrame matches the expected column names and data types.
        Removes any columns not defined in the schema.
        """
        # 1. Get the list of allowed columns
        schema_cols = list(self.config.SCHEMA_DATA_TYPES.keys())
        if 'match_date' not in schema_cols:
            schema_cols.append('match_date')

        # 2. Add missing columns and enforce types
        for col in schema_cols:
            if col not in df.columns:
                df[col] = np.nan
            
            dtype = self.config.SCHEMA_DATA_TYPES.get(col)
            if dtype:
                if dtype == "string":
                    df[col] = df[col].astype(str).replace(['nan', 'None', '<NA>', ''], pd.NA).astype(pd.StringDtype())
                elif dtype in ["Int64", "float64"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if dtype == "Int64":
                        df[col] = df[col].astype(pd.Int64Dtype())

        # 3. Final filtering (Removes 'Date' and 'Time')
        return df[schema_cols].copy()

    def _validate_critical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows that are missing essential information defined in config.
        """
        # Ensure we look for 'match_date' instead of the raw 'Date'
        check_list = [c if c != 'Date' else 'match_date' for c in self.config.CRITICAL_COLUMNS]
        
        # Only check columns that actually exist in the DataFrame
        valid_cols = [c for c in check_list if c in df.columns]
        
        initial_rows = len(df)
        df.dropna(subset=valid_cols, inplace=True)
        
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows due to missing critical data.")
            
        return df

    def map_teams_to_ids(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
        team_to_id_map = {str(team): i for i, team in enumerate(sorted(all_teams))}
        
        df['home_team_idx'] = df['HomeTeam'].astype(str).map(team_to_id_map).astype(pd.Int64Dtype())
        df['away_team_idx'] = df['AwayTeam'].astype(str).map(team_to_id_map).astype(pd.Int64Dtype())
        
        return df, team_to_id_map

    def get_market_odds_columns(self, df: pd.DataFrame, market_type: Market) -> Dict[str, List[str]]:
        patterns = self.config.ODDS_COL_PATTERNS.get(market_type, {})
        if not patterns:
            # Note: The test doesn't check this specific message, but let's keep it tidy
            raise ValueError(f"No patterns defined for market type: {market_type.name}")
            
        odds_cols = {}
        for period in ['pre_match', 'closing']:
            period_patterns = patterns.get(period, {})
            for provider, pattern in period_patterns.items():
                matched = [col for col in df.columns if re.search(pattern, col)]
                if matched:
                    odds_cols.setdefault(provider, []).extend(matched)
        
        # This is the part the test is checking:
        if not odds_cols:
            raise ValueError(f"No odds columns found for market type: {market_type.name} in the DataFrame.")
            
        return odds_cols
    
    def handle_inconsistent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if critical columns are completely missing from the DataFrame.
        Raises ValueError if a required column is not found in the headers.
        """
        # We need to account for 'Date' potentially being 'match_date'
        critical_to_check = self.config.CRITICAL_COLUMNS
        print(critical_to_check)
        for col in critical_to_check:
            print(col)
            print(df.columns)
            if col not in df.columns:
                # This EXACT string is required by the pytest 'match' argument
                raise ValueError(f"Critical column '{col}' is entirely missing")
                
        return df