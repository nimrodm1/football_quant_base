import pandas as pd
import io
from typing import List
from quant_football.core.config import DataConfig
from quant_football.data.preprocessor import Preprocessor
from quant_football.utils.logger import logger

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.preprocessor = Preprocessor(config)

    def _read_csv(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Reading file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            processed_content = content.replace('\t', ',')
            
            df = pd.read_csv(
                io.StringIO(processed_content),
                sep=None,
                engine='python',
                na_values=self.config.MISSING_VALUE_PLACEHOLDERS,
                skipinitialspace=True
            )
            df.columns = df.columns.str.replace('.', '_', regex=False)
            return df
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def load_dataset(self, file_paths: List[str]) -> pd.DataFrame:
        all_dfs = []
        for path in file_paths:
            df = self._read_csv(path)
            if not df.empty:
                all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Orchestrate the preprocessing steps exactly as the tests expect
        combined_df = self.preprocessor.clean_and_standardise(combined_df)
        
        return combined_df