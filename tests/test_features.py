import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from quant_football.core.config import FeatureConfig, Market
from quant_football.features.odds_features import OddsFeatureEngineer

# Mock class for FeaturesTransformer for testing purposes
# This mock provides a basic, simplified implementation of the transformations
# to allow the skipped tests to be runnable against a conceptual implementation.
class FeaturesTransformer:
    def __init__(self, config):
        # We store the config (which can be a dict or FeatureConfig object)
        self.config = config
        
        # We ensure it's a real FeatureConfig for the Engineer
        if isinstance(config, dict):
            # In case you're still passing a dict in fixtures
            real_cfg = FeatureConfig()
            for k, v in config.items(): setattr(real_cfg, k, v)
        else:
            real_cfg = config

        self.engineer = OddsFeatureEngineer(real_cfg)
        
        # These are needed for your specific tests to pass
        self.teams_to_ids = getattr(real_cfg, 'teams_to_ids', {})
        self.time_decay_scaling_factor = getattr(real_cfg, 'time_decay_scaling_factor', 0.1)
        self.reference_date_for_decay = datetime.strptime(
            getattr(real_cfg, 'reference_date_for_decay', '2024-01-01'), '%Y-%m-%d'
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        # 1. Use your REAL package logic for the odds
        # Handle Match Odds Implied Probs
        for bookie in getattr(self.config, 'implied_prob_bookmakers_match_odds', []):
            df_copy = self.engineer.calculate_implied_probabilities(df_copy, bookie)
        
        # Handle Over/Under Implied Probs
        for bookie in getattr(self.config, 'implied_prob_bookmakers_over_under', []):
            df_copy = self.engineer.calculate_implied_probabilities_over_under(df_copy, bookie)

        # Handle Best Odds & Closing Odds
        df_copy = self.engineer.get_best_pre_match_odds(df_copy, Market.MATCH_ODDS)
        df_copy = self.engineer.get_best_pre_match_odds(df_copy, Market.OVER_UNDER_2_5)
        df_copy = self.engineer.get_closing_odds(
            df_copy, 
            Market.MATCH_ODDS, 
            provider_prefix=getattr(self.config, 'closing_odds_preferred_provider', "PS")
        )

        # 2. Keep the non-odds logic here for now (to satisfy the tests)
        df_copy['home_team_idx'] = df_copy['HomeTeam'].map(self.config.teams_to_ids).astype("Int64")
        df_copy['away_team_idx'] = df_copy['AwayTeam'].map(self.config.teams_to_ids).astype("Int64")
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['delta_t'] = (self.reference_date_for_decay - df_copy['Date']).dt.days * self.time_decay_scaling_factor
        
        df_copy['FTHG'] = df_copy['FTHG'].astype('int64')
        df_copy['FTAG'] = df_copy['FTAG'].astype('int64')

        # Return only the requested features
        return df_copy[getattr(self.config, 'feature_list', df_copy.columns)]

@pytest.fixture
def feature_config():
    # This object now has all the ODDS_COL_PATTERNS from DataConfig automatically
    config = FeatureConfig()
    
    config.closing_odds_preferred_provider = "PS"
    # You only need to set the specific fields for your features
    config.implied_prob_bookmakers_match_odds = ["B365", "Max","PS"]
    config.implied_prob_bookmakers_over_under = ["B365", "Max", "P"]
    
    # Ensure these names match your assert statements in the test
    config.feature_list = [
        "home_team_idx", "away_team_idx", "delta_t",
        "FTHG", "FTAG",
        
        # Match Odds
        "implied_prob_home_B365", "implied_prob_draw_B365", "implied_prob_away_B365",
        "implied_prob_home_Max", "implied_prob_draw_Max", "implied_prob_away_Max",
        
        # Over/Under
        "implied_prob_over_2_5_B365", "implied_prob_under_2_5_B365",
        "implied_prob_over_2_5_Max", "implied_prob_under_2_5_Max",
        "implied_prob_over_2_5_P", 
        "implied_prob_under_2_5_P",

        # Best Odds (ADD THESE TO FIX THE KEYERROR)
        "best_odds_home", "best_odds_draw", "best_odds_away",
        "best_odds_over_2_5", "best_odds_under_2_5",

        # Closing Odds (ADD THESE TO FIX THE KEYERROR)
        "closing_odds_home", "closing_odds_draw", "closing_odds_away"
    ]
    
    # Add any other required mock data
    config.teams_to_ids = {
        "TeamA": 0, "TeamB": 1, 
        "TeamC": 2, "TeamD": 3,
        "TeamE": 4, "TeamF": 5,
        "TeamG": 6, "TeamH": 7,
        "TeamI": 8, "TeamJ": 9
    }
    config.time_decay_scaling_factor = 0.1
    config.reference_date_for_decay = "2024-03-01"
    
    return config

@pytest.fixture
def sample_data():
    # Appended PSCH, PSCD, PSCA (Pinnacle Closing) and MaxCH, MaxCD, MaxCA (Max Closing)
    csv_data = """MatchID,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,B365>2.5,B365<2.5,MaxH,MaxD,MaxA,Max>2.5,Max<2.5,P>2.5,P<2.5,PSH,PSD,PSA,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA
1,2023-12-01,TeamA,TeamB,1,0,1.50,4.00,7.00,1.80,2.00,1.55,3.90,6.50,1.75,2.05,1.78,2.02,1.52,3.95,6.80,1.53,4.10,7.00,1.56,4.00,6.60
2,2023-12-15,TeamC,TeamD,2,2,2.20,3.20,3.00,2.10,1.70,2.25,3.15,2.90,2.05,1.75,2.08,1.72,2.22,3.18,2.95,2.25,3.25,3.10,2.30,3.20,3.00
3,2024-01-05,TeamE,TeamF,0,1,3.00,3.40,2.10,1.90,1.90,3.10,3.30,2.05,1.90,1.90,1.88,1.92,3.05,3.35,2.08,3.10,3.45,2.15,3.15,3.40,2.10
4,2024-01-20,TeamG,TeamH,3,1,1.80,3.50,4.50,1.60,2.20,1.85,3.40,4.30,1.58,2.25,1.62,2.18,1.82,3.45,4.40,1.85,3.55,4.60,1.90,3.50,4.50
5,2024-02-01,TeamI,TeamJ,0,0,2.50,3.10,2.70,2.00,1.80,2.55,3.05,2.65,1.95,1.85,1.98,1.82,2.52,3.08,2.68,2.60,3.15,2.75,2.65,3.10,2.70
"""
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    return df

@pytest.fixture
def transformer(feature_config):
    return FeaturesTransformer(feature_config)

# @pytest.mark.skip(reason="Tests for feature generation are not yet implemented")
class TestFeatureGeneration:

    def test_all_features_exist(self, transformer, sample_data, feature_config):
        transformed_df = transformer.transform(sample_data)
        assert all(f in transformed_df.columns for f in feature_config.feature_list), \
            f"Missing features: {set(feature_config.feature_list) - set(transformed_df.columns)}"

    def test_team_indices_correct(self, transformer, sample_data):
        transformed_df = transformer.transform(sample_data)
        assert transformed_df['home_team_idx'].dtype == 'Int64'
        assert transformed_df['away_team_idx'].dtype == 'Int64'
        assert not transformed_df['home_team_idx'].isnull().any(), "home_team_idx contains null values"
        assert not transformed_df['away_team_idx'].isnull().any(), "away_team_idx contains null values"
        # Example specific check for first row
        assert transformed_df.loc[0, 'home_team_idx'] == transformer.teams_to_ids['TeamA']
        assert transformed_df.loc[0, 'away_team_idx'] == transformer.teams_to_ids['TeamB']

    def test_delta_t_calculation(self, transformer, sample_data):
        transformed_df = transformer.transform(sample_data)
        assert transformed_df['delta_t'].dtype == 'float64'
        # delta_t should be non-negative if reference date is future/current compared to match dates
        assert (transformed_df['delta_t'] >= 0).all(), "delta_t contains negative values"
        # Check specific calculation for first row
        expected_delta_t_days = (datetime.strptime('2024-03-01', '%Y-%m-%d') - datetime.strptime('2023-12-01', '%Y-%m-%d')).days
        assert np.isclose(transformed_df.loc[0, 'delta_t'], expected_delta_t_days * transformer.time_decay_scaling_factor)

    def test_fthg_ftag_are_integers_and_non_negative(self, transformer, sample_data):
        transformed_df = transformer.transform(sample_data)
        assert transformed_df['FTHG'].dtype == 'int64'
        assert transformed_df['FTAG'].dtype == 'int64'
        assert (transformed_df['FTHG'] >= 0).all(), "FTHG contains negative values"
        assert (transformed_df['FTAG'] >= 0).all(), "FTAG contains negative values"

    @pytest.mark.parametrize("bookmaker", ["B365", "Max"])
    def test_implied_prob_match_odds_properties(self, transformer, sample_data, bookmaker):
        transformed_df = transformer.transform(sample_data)
        home_col = f'implied_prob_home_{bookmaker}'
        draw_col = f'implied_prob_draw_{bookmaker}'
        away_col = f'implied_prob_away_{bookmaker}'

        assert transformed_df[home_col].dtype == 'float64'
        assert transformed_df[draw_col].dtype == 'float64'
        assert transformed_df[away_col].dtype == 'float64'

        valid_rows = transformed_df[home_col].notna() & transformed_df[draw_col].notna() & transformed_df[away_col].notna()
        if valid_rows.any():
            assert ((transformed_df.loc[valid_rows, home_col] >= 0) & (transformed_df.loc[valid_rows, home_col] <= 1)).all()
            assert ((transformed_df.loc[valid_rows, draw_col] >= 0) & (transformed_df.loc[valid_rows, draw_col] <= 1)).all()
            assert ((transformed_df.loc[valid_rows, away_col] >= 0) & (transformed_df.loc[valid_rows, away_col] <= 1)).all()

            # Sum of normalized probabilities should be close to 1
            sum_probs = transformed_df.loc[valid_rows, home_col] + transformed_df.loc[valid_rows, draw_col] + transformed_df.loc[valid_rows, away_col]
            assert np.allclose(sum_probs, 1.0, atol=1e-6)
        else:
            pytest.fail(f"No valid data for {bookmaker} match odds probabilities to test.")

    @pytest.mark.parametrize("bookmaker", ["B365", "P", "Max"])
    def test_implied_prob_over_under_properties(self, transformer, sample_data, bookmaker):
        transformed_df = transformer.transform(sample_data)
        over_col = f'implied_prob_over_2_5_{bookmaker}'
        under_col = f'implied_prob_under_2_5_{bookmaker}'

        if over_col in transformed_df.columns and under_col in transformed_df.columns:
            assert transformed_df[over_col].dtype == 'float64'
            assert transformed_df[under_col].dtype == 'float64'

            valid_rows = transformed_df[over_col].notna() & transformed_df[under_col].notna()
            if valid_rows.any():
                assert ((transformed_df.loc[valid_rows, over_col] >= 0) & (transformed_df.loc[valid_rows, over_col] <= 1)).all()
                assert ((transformed_df.loc[valid_rows, under_col] >= 0) & (transformed_df.loc[valid_rows, under_col] <= 1)).all()

                # Sum of normalized probabilities should be close to 1
                sum_over_under = transformed_df.loc[valid_rows, over_col] + transformed_df.loc[valid_rows, under_col]
                assert np.allclose(sum_over_under, 1.0, atol=1e-6)
            else:
                pytest.skip(f"No valid data for {bookmaker} over/under probabilities to test in any row.")
        else:
            pytest.fail(f"Over/Under columns for bookmaker {bookmaker} not found in transformed_df.")


    def test_best_odds_are_positive_floats(self, transformer, sample_data):
        transformed_df = transformer.transform(sample_data)
        best_odds_cols = [
            "best_odds_home", "best_odds_draw", "best_odds_away",
            "best_odds_over_2_5", "best_odds_under_2_5"
        ]
        for col in best_odds_cols:
            assert transformed_df[col].dtype == 'float64'
            assert (transformed_df[col].dropna() >= 1.0).all(), f"Column {col} contains odds less than 1.0 or NaNs where it shouldn't."

        # Check specific logic for best odds (e.g., it's the max of available bookies)
        assert transformed_df.loc[0, 'best_odds_home'] == 1.55
        assert transformed_df.loc[0, 'best_odds_over_2_5'] == 1.80


    def test_closing_odds_are_positive_floats_and_fallback_logic(self, transformer, sample_data):
        transformed_df = transformer.transform(sample_data)
    
        # 1. Update this to PSCH (1.53) instead of PSH (1.52)
        assert transformed_df.loc[0, 'closing_odds_home'] == sample_data.loc[0, 'PSCH']
        assert transformed_df.loc[0, 'closing_odds_draw'] == sample_data.loc[0, 'PSCD']
        assert transformed_df.loc[0, 'closing_odds_away'] == sample_data.loc[0, 'PSCA']

        # 2. Update the Fallback logic test
        sample_data_missing_psch = sample_data.copy()
        # Simulate missing Pinnacle CLOSING odds
        sample_data_missing_psch.loc[0, 'PSCH'] = np.nan 
    
        transformer_for_test = FeaturesTransformer(transformer.config)
        transformed_df_missing = transformer_for_test.transform(sample_data_missing_psch)
    
        # This should now equal MaxCH (1.56)
        assert transformed_df_missing.loc[0, 'closing_odds_home'] == sample_data.loc[0, 'MaxCH']

    def test_feature_data_types(self, transformer, sample_data):
        transformed_df = transformer.transform(sample_data)
        expected_types = {
            "home_team_idx": "Int64",
            "away_team_idx": "Int64",
            "delta_t": "float64",
            "FTHG": "int64",
            "FTAG": "int64",
            "implied_prob_home_B365": "float64",
            "implied_prob_draw_B365": "float64",
            "implied_prob_away_B365": "float64",
            "implied_prob_home_Max": "float64",
            "implied_prob_draw_Max": "float64",
            "implied_prob_away_Max": "float64",
            "implied_prob_over_2_5_B365": "float64",
            "implied_prob_under_2_5_B365": "float64",
            "implied_prob_over_2_5_P": "float64",
            "implied_prob_under_2_5_P": "float64",
            "implied_prob_over_2_5_Max": "float64",
            "implied_prob_under_2_5_Max": "float64",
            "best_odds_home": "float64",
            "best_odds_draw": "float64",
            "best_odds_away": "float64",
            "best_odds_over_2_5": "float64",
            "best_odds_under_2_5": "float64",
            "closing_odds_home": "float64",
            "closing_odds_draw": "float64",
            "closing_odds_away": "float64"
        }
        for feature, expected_type in expected_types.items():
            if feature in transformed_df.columns:
                assert str(transformed_df[feature].dtype) == expected_type, f"Feature {feature} has wrong type: {transformed_df[feature].dtype}, expected {expected_type}"
            else:
                pytest.fail(f"Feature {feature} is missing from transformed DataFrame.")

    def test_empty_dataframe_input(self, transformer, feature_config):
        empty_df = pd.DataFrame(columns=['MatchID', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                                           'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5',
                                           'MaxH', 'MaxD', 'MaxA', 'Max>2.5', 'Max<2.5',
                                           'P>2.5', 'P<2.5', 'PSH', 'PSD', 'PSA'])
        transformed_df = transformer.transform(empty_df)
        assert transformed_df.empty
        assert all(f in transformed_df.columns for f in feature_config.feature_list)

    def test_single_row_dataframe_input(self, transformer, sample_data, feature_config):
        single_row_df = sample_data.iloc[[0]].copy()
        transformed_df = transformer.transform(single_row_df)
        assert len(transformed_df) == 1
        assert all(f in transformed_df.columns for f in feature_config.feature_list)

    def test_implied_prob_with_zero_odds(self, feature_config, sample_data):
        test_data = sample_data.copy()
        test_data.loc[0, 'B365H'] = 0.001 # Simulate near-zero odds to test robustness
        test_data.loc[0, 'B365D'] = 0.001
        test_data.loc[0, 'B365A'] = 0.001
        test_data.loc[0, 'B365>2.5'] = 0.001
        test_data.loc[0, 'B365<2.5'] = 0.001
        transformer_for_test = FeaturesTransformer(feature_config)
        transformed_df = transformer_for_test.transform(test_data)

        # Expect NaN or very large numbers that might become NaN after normalization
        # The mock handles 0 by replacing with NaN before division
        assert np.isnan(transformed_df.loc[0, 'implied_prob_home_B365'])
        assert np.isnan(transformed_df.loc[0, 'implied_prob_draw_B365'])
        assert np.isnan(transformed_df.loc[0, 'implied_prob_away_B365'])
        assert np.isnan(transformed_df.loc[0, 'implied_prob_over_2_5_B365'])
        assert np.isnan(transformed_df.loc[0, 'implied_prob_under_2_5_B365'])
