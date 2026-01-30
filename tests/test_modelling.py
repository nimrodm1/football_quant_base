import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
import pandas as pd
import numpy as np

import arviz as az
import warnings

from quant_football.modelling import BayesianPoissonGLMM, MatchPrediction
from quant_football.core.config import ModellingConfig, Market
import pymc as pm

import warnings
warnings.filterwarnings(
    "ignore", 
    message="The effect of Potentials on other parameters is ignored"
)


@pytest.fixture
def mock_data():
    data = [
        ["Arsenal", "Chelsea", 2, 1, "01/01/2023"],
        ["Liverpool", "Man City", 3, 2, "02/01/2023"],
        ["Chelsea", "Liverpool", 0, 2, "03/01/2023"],
        ["Man City", "Arsenal", 1, 1, "04/01/2023"],
        ["Tottenham", "Man Utd", 2, 2, "05/01/2023"],
        ["Man Utd", "Tottenham", 1, 0, "06/01/2023"],
        ["Arsenal", "Man Utd", 3, 1, "07/01/2023"],
        ["Chelsea", "Man City", 1, 1, "08/01/2023"],
    ]
    return pd.DataFrame(data, columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"])

@pytest.fixture
def fitted_model(mock_data):
    # 1. Clean the mock data to match the expected format
    # The model expects 'home_team_idx', 'away_team_idx', and 'match_date'
    df = mock_data.copy()
    df['match_date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 2. Create the mapping manually (simulating the DataLoader/Preprocessor agent)
    teams = sorted(list(set(df['HomeTeam']) | set(df['AwayTeam'])))
    mapping = {team: i for i, team in enumerate(teams)}
    
    # 3. Add the required index columns
    df['home_team_idx'] = df['HomeTeam'].map(mapping)
    df['away_team_idx'] = df['AwayTeam'].map(mapping)
    
    config = ModellingConfig()
    # Minimising sampling for test speed
    config.sampling = {"draws": 100, "tune": 100, "chains": 1, "random_seed": 42}
    
    model = BayesianPoissonGLMM(config=vars(config))
    
    # 4. Pass the data AND the mapping
    model.fit(df, teams_mapping=mapping)
    
    return model

def test_sum_to_zero_constraint(fitted_model):
    """Verify that att and defs parameters strictly satisfy sum-to-zero constraint."""
    post = fitted_model.trace.posterior
    att_mean = post["att"].mean(dim=["chain", "draw"]).values
    defs_mean = post["defs"].mean(dim=["chain", "draw"]).values
    # pm.ZeroSumNormal ensures sum is zero
    assert abs(np.sum(att_mean)) < 1e-10
    assert abs(np.sum(defs_mean)) < 1e-10

def test_predict_probabilities_format(fitted_model):
    # 1. Arrange: Create the input DataFrame with your new match_id
    matches = pd.DataFrame([
        {"match_id": "m1", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"},
        {"match_id": "m2", "HomeTeam": "Man City", "AwayTeam": "Liverpool"}
    ])
    
    # 2. Act: Call the model
    predictions = fitted_model.predict_outcome_probabilities(matches)
    
    # 3. Assert: Check the DOMAIN OBJECTS, not the DataFrame
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert isinstance(predictions[0], MatchPrediction)
    
    # Verify the ID was preserved
    assert predictions[0].match_id == "m1"
    
    # Check probability maths for one market
    match_odds = predictions[0].probabilities[Market.MATCH_ODDS]
    total_prob = sum(match_odds.values())
    assert pytest.approx(total_prob) == 1.0
    
def test_model_structure(fitted_model):
    """Verify that the model variables exist and have the correct dimensions."""
    vars_in_trace = fitted_model.trace.posterior.data_vars
    
    # Check for core parameters
    assert "mu" in vars_in_trace
    assert "att" in vars_in_trace
    assert "defs" in vars_in_trace
    assert "alpha" in vars_in_trace
    
    # Check coordinates (n_teams)
    # This ensures your 'att' vector is the right length for your league
    assert vars_in_trace["att"].sizes["att_dim_0"] == fitted_model.n_teams

def test_prediction_output_logic(fitted_model, mock_data):
    """Verify that the model produces logically sound goal expectations."""
    matches = mock_data[['HomeTeam', 'AwayTeam']].head(2)
    
    with fitted_model.model:
        # Changed "theta_home_det" to "theta_h_det" to match the class implementation
        ppc = pm.sample_posterior_predictive(fitted_model.trace, var_names=["theta_h_det"])
        
        # Verify the deterministic lambda is positive
        assert (ppc.posterior_predictive["theta_h_det"] > 0).all()

def test_directory_integrity():
    """QA Check: Directory Integrity."""
    assert os.path.exists("src/quant_football/"), "The package reside in src/"
    assert not os.path.exists("quant_football/"), "Regression: quant_football directory found in root"

def test_alpha_presence(fitted_model):
    """Verify time-decay parameter alpha is included in the model trace."""
    assert "alpha" in fitted_model.trace.posterior
