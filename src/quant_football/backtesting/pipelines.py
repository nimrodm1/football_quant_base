import pandas as pd
from typing import Optional, Dict, Any, List
from quant_football.data.data_loader import DataLoader
from quant_football.data.preprocessor import Preprocessor
from quant_football.core.config import DataConfig
from quant_football.modelling.base_model import BaseModel, MatchPrediction

class DataPipeline:
    def __init__(self, config: DataConfig):
        self.config = config
        # Use positional argument to be absolutely sure
        self.loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw data and runs it through the preprocessor.
        """
        return self.preprocessor.clean_and_standardise(data)

class ModelPipeline:
    def __init__(self, model: BaseModel):
        self.model = model

    def train(self, training_data: pd.DataFrame):
        """
        Trains the underlying model.
        """
        self.model.fit(training_data)

    def predict(self, home_team: str, away_team: str, **kwargs) -> List[MatchPrediction]:
        """
        Generates predictions for a single match.
        """
        return self.model.predict_outcome_probabilities(home_team, away_team, **kwargs)
