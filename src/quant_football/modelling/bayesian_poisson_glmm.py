import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from typing import Dict, Any, Optional, Tuple
from .base_model import BaseModel, MatchPrediction
from quant_football.utils.logger import logger
from quant_football.core.config import Market

class BayesianPoissonGLMM(BaseModel):
    """
    Bayesian Hierarchical Poisson GLMM for football match prediction.
    Features:
    - Hierarchical attack and defense parameters with ZeroSumNormal priors.
    - Time-weighted likelihood using pm.Potential and vectorized log-probability.
    - Standardized Prediction API utilizing pm.Data for out-of-sample inference.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.teams_mapping = {}
        self.inv_teams_mapping = {}
        self.n_teams = 0
        self.model = None
        self.trace = None

    def _prepare_data_indices(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts pre-mapped indices and calculates time decay values.
        Assumes clean_and_standardise has already been run.
        """
        # 1. Direct Extraction (Will throw KeyError if Preprocessor failed)
        h_idx = data['home_team_idx'].values
        a_idx = data['away_team_idx'].values
        h_goals = data['FTHG'].values
        a_goals = data['FTAG'].values
        match_dates = data['match_date']

        # 2. Global Indexing Size
        # We use the mapping stored in the model instance to define the vector shape
        self.n_teams = len(self.teams_mapping)

        # 3. Time Decay (Delta T)
        # Calculated as months (30-day blocks) from the most recent match in the dataset
        last_date = match_dates.max()
        delta_t = (last_date - match_dates).dt.days.values / 30.0
    
        return (
            h_idx.astype(int), 
            a_idx.astype(int), 
            h_goals.astype(int), 
            a_goals.astype(int), 
            delta_t
        )
    
    def _build_model(self, h_idx: np.ndarray, a_idx: np.ndarray, 
                     h_goals: np.ndarray, a_goals: np.ndarray, 
                     delta_t: np.ndarray):
        priors = self.config.get('priors', {})
        
        with pm.Model() as model:
            # Shared data containers
            h_team_obs = pm.Data("h_team_obs", h_idx)
            a_team_obs = pm.Data("a_team_obs", a_idx)
            dt_obs = pm.Data("dt_obs", delta_t)
            h_goals_obs = pm.Data("h_goals_obs", h_goals)
            a_goals_obs = pm.Data("a_goals_obs", a_goals)
            
            # Global parameters
            mu = pm.Normal("mu", 
                           mu=priors.get('mu', {}).get('mean', 0.0), 
                           sigma=priors.get('mu', {}).get('sd', 1.0))
            h_adv = pm.Normal("h_adv", 
                              mu=priors.get('h_adv', {}).get('mean', 0.3), 
                              sigma=priors.get('h_adv', {}).get('sd', 0.2))
            
            sigma_att = pm.HalfNormal("sigma_att", sigma=priors.get('sigma_att', {}).get('sd', 0.5))
            sigma_def = pm.HalfNormal("sigma_def", sigma=priors.get('sigma_def', {}).get('sd', 0.5))
            
            # Team effects with sum-to-zero constraint
            att = pm.ZeroSumNormal("att", sigma=sigma_att, shape=self.n_teams)
            defs = pm.ZeroSumNormal("defs", sigma=sigma_def, shape=self.n_teams)
            
            # Time decay hyperparameter
            alpha = pm.HalfNormal("alpha", sigma=priors.get('alpha', {}).get('sd', 0.1))
            
            # Expected goal rates (log-link)
            theta_h = pm.math.exp(mu + h_adv + att[h_team_obs] - defs[a_team_obs])
            theta_a = pm.math.exp(mu + att[a_team_obs] - defs[h_team_obs])
            
            # Weighted Log-Likelihood via pm.Potential
            h_dist = pm.Poisson.dist(mu=theta_h)
            a_dist = pm.Poisson.dist(mu=theta_a)
            h_logp = pm.logp(h_dist, h_goals_obs)
            a_logp = pm.logp(a_dist, a_goals_obs)
            
            decay_weights = pm.math.exp(-alpha * dt_obs)
            pm.Potential("weighted_logp", (h_logp + a_logp) * decay_weights)
            
            # Track rates for posterior prediction
            pm.Deterministic("theta_h_det", theta_h)
            pm.Deterministic("theta_a_det", theta_a)
            
        return model

    def fit(self, data: pd.DataFrame, teams_mapping: Dict[str, int], **kwargs):
        logger.info(f"Fitting BayesianPoissonGLMM with {len(data)} matches.")
        self.teams_mapping = teams_mapping
        self.inv_teams_mapping = {v: k for k, v in teams_mapping.items()}
        h_idx, a_idx, h_goals, a_goals, delta_t = self._prepare_data_indices(data)
        
        self.model = self._build_model(h_idx, a_idx, h_goals, a_goals, delta_t)
        sampling_params = self.config.get('sampling', {})

        with self.model:
            self.trace = pm.sample(
                draws=sampling_params.get('draws', 2000),
                tune=sampling_params.get('tune', 1000),
                chains=sampling_params.get('chains', 4),
                target_accept=sampling_params.get('target_accept', 0.95),
                random_seed=sampling_params.get('random_seed', 42),
                **kwargs
            )
        return self.trace

    def predict_outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outcomes for matches.
        Returns DataFrame with ['home_win', 'draw', 'away_win', 'over_2_5', 'under_2_5']
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction.")

        h_idx = matches['HomeTeam'].map(self.teams_mapping).values
        a_idx = matches['AwayTeam'].map(self.teams_mapping).values
        
        if np.isnan(h_idx).any() or np.isnan(a_idx).any():
            missing = set(matches['HomeTeam'][np.isnan(h_idx)]) | set(matches['AwayTeam'][np.isnan(a_idx)])
            raise ValueError(f"Encountered unknown teams during prediction: {missing}")

        h_idx = h_idx.astype(int)
        a_idx = a_idx.astype(int)
        
        dummy_goals = np.zeros(len(matches), dtype=int)
        
        with self.model:
            pm.set_data({
                "h_team_obs": h_idx,
                "a_team_obs": a_idx,
                "dt_obs": np.zeros(len(matches)),
                "h_goals_obs": dummy_goals,
                "a_goals_obs": dummy_goals
            })
            ppc = pm.sample_posterior_predictive(self.trace, var_names=["theta_h_det", "theta_a_det"])
            
        l_h = ppc.posterior_predictive["theta_h_det"].values.reshape(-1, len(matches))
        l_a = ppc.posterior_predictive["theta_a_det"].values.reshape(-1, len(matches))
        
        n_samples_conf = self.config.get('prediction', {}).get('n_samples', 4000)
        indices = np.random.choice(l_h.shape[0], size=min(n_samples_conf, l_h.shape[0]), replace=False)
        
        sim_h = np.random.poisson(l_h[indices, :])
        sim_a = np.random.poisson(l_a[indices, :])
        
        predictions = []
        for i, (_, row) in enumerate(matches.iterrows()):
            h_s = sim_h[:, i]
            a_s = sim_a[:, i]
            
            probs = {
                Market.MATCH_ODDS: {
                    "home": float(np.mean(h_s > a_s)),
                    "draw": float(np.mean(h_s == a_s)),
                    "away": float(np.mean(h_s < a_s))
                },
                Market.OVER_UNDER_2_5: {
                    "over": float(np.mean((h_s + a_s) > 2.5)),
                    "under": float(np.mean((h_s + a_s) < 2.5))
                }
            }
            
            predictions.append(MatchPrediction(
                match_id=str(row['match_id']),
                home_team=row['HomeTeam'],
                away_team=row['AwayTeam'],
                probabilities=probs
            ))
        return predictions

    def save(self, path: str):
        if self.trace is not None:
            self.trace.to_netcdf(path)

    def load(self, path: str, training_data: pd.DataFrame):
        self.trace = az.from_netcdf(path)
        h, a, h_g, a_g, dt = self._prepare_data_indices(training_data)
        self.model = self._build_model(h, a, h_g, a_g, dt)
