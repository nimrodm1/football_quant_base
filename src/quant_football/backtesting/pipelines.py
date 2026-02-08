import pandas as pd
from typing import Optional, Dict, Any, List
from quant_football.data.data_loader import DataLoader
from quant_football.core.config import DataConfig
from quant_football.modelling.base_model import BaseModel, MatchPrediction

class DataPipeline:
    def __init__(self, config: DataConfig):
        self.config = config
        self.loader = DataLoader(config)

    def prepare_data(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Orchestrates the loading and standardisation of data.
        Raises ValueError if no data is found or processed.
        """
        processed_df = self.loader.load_dataset(file_paths)
        
        if processed_df.empty:
            raise ValueError(
                f"Data preparation failed: No data could be loaded from {file_paths}. "
                "Check file paths and standardisation logic."
            )
            
        return processed_df

    @property
    def teams_mapping(self) -> Dict[str, Any]:
        """
        Access the team mappings captured by the loader.
        """
        return self.loader.teams_mapping

class ModelPipeline:
    def __init__(self, model: BaseModel, sampler_config: Optional[Dict[str, Any]] = None):
        self.model = model
        # Default to your preferred high-performance config
        self.sampler_config = sampler_config or {
            "nuts_sampler": "nutpie",
            "mode": "NUMBA"
        }

    def train(self, training_data: pd.DataFrame, teams_mapping: Dict[str, int]):
        """
        Executes training using the global team mapping and fast sampler.
        """
        self.model.fit(
            training_data, 
            teams_mapping=teams_mapping, 
            **self.sampler_config
        )

    def predict(self, matches: pd.DataFrame, **kwargs) -> List[MatchPrediction]:
        """
        Generates predictions for matches in the provided DataFrame.
        Expects DataFrame with columns: HomeTeam, AwayTeam, match_id
        """
        return self.model.predict_outcome_probabilities(matches, **kwargs)
