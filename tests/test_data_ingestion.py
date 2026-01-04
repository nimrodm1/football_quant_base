import pytest
import pandas as pd
import io
from datetime import datetime

from quant_football.core.config import DataConfig, Market
from quant_football.data.data_loader import DataLoader
from quant_football.data.preprocessor import Preprocessor

class TestDataIngestion:

    @pytest.fixture
    def mock_config(self):
        # Create an instance of DataConfig
        return DataConfig()

    @pytest.fixture
    def mock_data(self):
        # Your provided mock CSV data
        return """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365.H,B365.D,B365.A,B365.2.5,B365.U2.5,AHh,B365AHH,B365AHA,Max.H,Max.D,Avg.H,Avg.D,P.2.5,P.U2.5,AHCh,AvgCAHH,AvgCAHA
E0,01/08/21,15:00,Man United,Liverpool,2,1,H,1,0,H,M Dean,10,8,5,3,12,15,6,4,1,2,0,0,1.80,3.50,4.20,1.60,2.20,0.5,1.90,1.90,2.00,3.75,1.75,3.40,1.65,2.15,0.75,2.00,1.80
E0,02/08/2021,17:30,Arsenal FC,Chelsea,0,0,D,0,0,D,,8,6,3,2,10,11,5,3,0,1,0,0,NA,NA,NA,1.80,2.00,0.0,NA,NA,NA,NA,1.70,3.50,NA,NA,0.0,2.10,1.70
E1,03/09/21,20:00,Leicester,Tottenham,1,1,D,1,1,D,A Taylor,,NA,6,4,14,12,7,5,2,1,0,0,2.50,3.20,2.80,NA,NA,0.25,1.95,1.85,2.60,3.30,2.40,3.10,1.70,2.10,0.5,NA,NA
E0,04/09/2021,15:00,Man United,Everton,1,1,D,1,0,H,M Dean,12,10,4,4,11,13,8,6,2,1,0,0,2.00,3.30,3.70,1.70,2.10,0.5,1.85,1.95,2.10,3.50,1.90,3.20,1.75,2.05,0.75,1.90,1.90
E0	05/10/21,15:00,Wolves,Southampton,3,0,H,1,0,H,P Tierney,15,7,8,2,10,9,7,3,1,1,0,0,2.10,3.20,3.50,1.90,1.90,-0.25,1.80,2.00,2.20,3.40,2.00,3.10,1.95,1.85,-0.5,1.75,2.05
E0,06/11/2021,12:30,Leeds,,1,2,A,1,1,D,,14,9,7,5,15,10,8,4,3,2,0,0,1.90,3.40,3.80,1.75,2.05,0.0,1.90,1.90,2.00,3.60,1.85,3.30,1.80,2.00,0.0,1.95,1.85
E0,07/12/21,15:00,Aston Villa,West Ham,2,2,D,1,1,D,J Moss,11,11,6,6,13,13,5,5,1,1,0,0,2.30,3.10,3.00,1.80,2.00,0.25,1.90,1.90,2.40,3.20,2.20,3.00,1.85,1.95,0.0,1.90,1.90
E0,08/01/2022,17:00,Newcastle,Brighton,NA,NA,D,0,0,D,S Attwell,9,10,3,4,10,11,4,5,2,1,0,0,2.60,3.20,2.70,1.90,1.90,0.0,1.90,1.90,2.70,3.30,2.50,3.10,1.95,1.85,0.0,1.90,1.90
E0,09/02/22,20:00,Liverpool,Man City,1,1,D,1,0,H,A Marriner,10,12,4,6,12,10,6,7,1,2,0,0,3.00,3.40,2.30,1.70,2.10,0.75,1.85,1.95,3.10,3.60,2.90,3.30,1.75,2.05,0.5,1.80,2.00
E0,10/03/2022,15:00,Tottenham,Arsenal,NA,NA,D,NA,NA,D,P Tierney,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2.20,3.20,3.20,1.80,2.00,0.0,1.90,1.90,2.30,3.30,2.10,3.10,1.85,1.95,0.0,1.90,1.90
E0,11/04/22,17:30,West Ham,Aston Villa,1,0,H,0,0,D,M Dean,10,9,4,3,11,12,5,4,1,1,0,0,2.00,3.30,3.70,1.65,2.15,0.5,1.85,1.95,2.10,3.50,1.90,3.20,1.70,2.10,0.75,1.90,1.90
"""

    # Helper function to simulate the full ingestion process using the actual classes
    def full_ingestion_process(self, raw_csv_content: str, config: DataConfig) -> pd.DataFrame:
        dummy_file_path = "mock_data.csv"
        with open(dummy_file_path, "w", encoding='utf-8') as f:
            f.write(raw_csv_content)

        data_loader = DataLoader(config)
        preprocessor = Preprocessor(config) # We need this for the mapping method

    # 1. Load, Clean, and Standardise (Internal to DataLoader)
        df = data_loader.load_dataset([dummy_file_path])
    
    # 2. Add the specific features the tests expect (Indices)
    # This creates the 'home_team_idx' and 'away_team_idx' columns
        df, _ = preprocessor.map_teams_to_ids(df)
    
    # Clean up
        import os
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

        return df

    def test_ingest_data_mock_check(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # 12 initial rows. Row 8 (Newcastle FTHG/FTAG NA) and Row 10 (Tottenham FTHG/FTAG NA) are dropped.
        # This leaves 10 rows.
        assert len(df) == 8, f"Mock ingestion dropped {11 - len(df)} rows, expected 2 rows dropped. Actual: {len(df)}"

    def test_all_expected_columns_present(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        expected_columns = list(mock_config.SCHEMA_DATA_TYPES.keys())
        # The ingestion logic should ensure all expected columns are present, even if some are entirely NA.
        # Added 'match_date', 'home_team_idx', 'away_team_idx' that are created during preprocessing
        expected_columns.extend(['match_date', 'home_team_idx', 'away_team_idx'])
        
        # Remove 'Date' and 'Time' as they are replaced by 'match_date'
        if 'Date' in expected_columns:
            expected_columns.remove('Date')
        if 'Time' in expected_columns:
            expected_columns.remove('Time')

        assert set(df.columns) == set(expected_columns), f"Final DataFrame columns do not match expected schema. Missing: {set(expected_columns) - set(df.columns)}, Extra: {set(df.columns) - set(expected_columns)}"

    def test_column_renaming(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # Check specific mapped columns derived from dot-separated raw names
        assert "B365H" in df.columns
        assert "B365>2.5" in df.columns
        assert "MaxH" in df.columns
        assert "AvgH" in df.columns
        assert "P>2.5" in df.columns
        assert "AHCh" in df.columns
        assert "AvgCAHH" in df.columns
        # Ensure original dotted columns (e.g., B365.H) are no longer present
        assert "B365.H" not in df.columns
        assert "Max.H" not in df.columns
        assert "B365_H" not in df.columns # Ensure underscore-standardized raw names are also gone after mapping

    def test_date_parsing_and_format(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        assert 'match_date' in df.columns
        assert pd.api.types.is_datetime64_ns_dtype(df['match_date'])

        # Create the target date as a pandas Timestamp for a clean comparison
        target_date = pd.Timestamp(2021, 8, 1, 15, 0)
        
        # Check if the value exists in the column
        assert (df['match_date'] == target_date).any(), f"Expected {target_date} to be in match_date column."

    def test_time_parsing_and_format(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # 'Time' column is dropped after conversion to 'match_date'
        assert 'Time' not in df.columns, "'Time' column should be dropped after processing."

    def test_missing_value_handling(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # Check Referee column which has empty strings and 'None' in mock data
        # Note: Referee is not a critical column, so NaNs should persist
        assert df['Referee'].isnull().any(), "'Referee' column should contain missing values."
        # Check B365H (originally NA) for Arsenal-Chelsea match
        # Note: After preprocessor converts to float, NA becomes NaN
        arsenal_chelsea_row = df[(df['HomeTeam'] == 'Arsenal') & (df['AwayTeam'] == 'Chelsea')]
        assert not arsenal_chelsea_row.empty, "Arsenal-Chelsea row should exist."
        assert pd.isna(arsenal_chelsea_row['B365H'].iloc[0]), "'B365H' for Arsenal-Chelsea should be NA/NaN."
        
        # Check HS (originally empty) for Leicester-Tottenham match
        leicester_tottenham_row = df[(df['HomeTeam'] == 'Leicester City') & (df['AwayTeam'] == 'Tottenham')]
        assert not leicester_tottenham_row.empty, "Leicester-Tottenham row should exist."
        assert pd.isna(leicester_tottenham_row['HS'].iloc[0]), "'HS' for Leicester-Tottenham should be NA/NaN."


    def test_data_type_conversion(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        for col, expected_dtype_str in mock_config.SCHEMA_DATA_TYPES.items():
            if col in df.columns: # Check only for columns that are expected to be in the final DF
                if expected_dtype_str == "string":
                    assert pd.api.types.is_string_dtype(df[col]), f"Column '{col}' expected to be string, got {df[col].dtype}."
                elif expected_dtype_str == "Int64": # Pandas nullable integer type
                    assert str(df[col].dtype) == "Int64", f"Column '{col}' expected to be Int64, got {df[col].dtype}."
                elif expected_dtype_str == "float64":
                    assert pd.api.types.is_float_dtype(df[col]), f"Column '{col}' expected to be float64, got {df[col].dtype}."
                else:
                    pytest.fail(f"Unhandled dtype '{expected_dtype_str}' in test for column '{col}'.")
            elif col in ['Date', 'Time'] and 'match_date' in df.columns:
                 # 'Date' and 'Time' are expected to be removed and replaced by 'match_date'
                 pass
            elif col not in ['match_date', 'home_team_idx', 'away_team_idx']: # These are added later, not directly from schema
                pytest.fail(f"Column '{col}' from schema not found in final DataFrame.")
        
        # Additionally check the dtypes of the newly created columns
        assert pd.api.types.is_datetime64_ns_dtype(df['match_date']), "'match_date' expected to be datetime."
        assert str(df['home_team_idx'].dtype) == 'Int64', "'home_team_idx' expected to be Int64."
        assert str(df['away_team_idx'].dtype) == 'Int64', "'away_team_idx' expected to be Int64."


    def test_team_name_standardization(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # Verify specific team name replacements
        assert "Man Utd" in df['HomeTeam'].values
        assert "Arsenal" in df['HomeTeam'].values
        assert "Leicester City" in df['HomeTeam'].values
        # Ensure original names are no longer present
        assert "Man United" not in df['HomeTeam'].values
        assert "Arsenal FC" not in df['HomeTeam'].values
        assert "Leicester" not in df['HomeTeam'].values

    def test_critical_columns_integrity(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # Adjust critical columns for checking, as 'Date' is replaced by 'match_date'
        effective_critical_columns = []
        for col in mock_config.CRITICAL_COLUMNS:
            if col == 'Date':
                effective_critical_columns.append('match_date')
            else:
                effective_critical_columns.append(col)

        for col in effective_critical_columns:
            if col in df.columns: # Only check columns that are actually present
                assert not df[col].isnull().any(), f"Critical column '{col}' contains missing values after critical row dropping."
            else:
                pytest.fail(f"Critical column '{col}' is missing from the DataFrame.")

        # Verify the total number of rows after dropping those with missing critical data
        # 11 initial rows -> 3 rows (Leeds-Brentford, Newcastle-Brighton, Tottenham-Arsenal) dropped due to away team and FTHG/FTAG being NA.
        assert len(df) == 8, "Incorrect number of rows after dropping critical missing data."

        # Test the ValueError for entirely missing critical column
        # Modify config to make a non-existent column critical
        temp_config = DataConfig()
        temp_config.CRITICAL_COLUMNS = ["NonExistentCriticalCol"]
        preprocessor = Preprocessor(temp_config)
        # Expect ValueError when handling inconsistent columns
        with pytest.raises(ValueError, match="Critical column 'NonExistentCriticalCol' is entirely missing"):
            # Use a dummy dataframe for this specific test
            dummy_df = pd.DataFrame({'Div': ['E0']})
            preprocessor.handle_inconsistent_columns(dummy_df)


    def test_odds_column_typing(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # Check a representative sample of odds columns
        odds_cols_to_check = ["B365H", "B365>2.5", "B365AHH", "MaxH", "AvgH", "P>2.5", "AHCh", "AvgCAHH"]
        for col in odds_cols_to_check:
            if col in df.columns:
                assert pd.api.types.is_float_dtype(df[col]), f"Odds column '{col}' expected to be float64, got {df[col].dtype}."
                # Handicaps (AHh, AHCh) can be negative, other odds should generally be positive
                if col not in ["AHh", "AHCh"]:
                    assert (df[col].dropna() >= 0).all(), f"Odds column '{col}' contains non-positive values (excluding NaN)."
            else:
                pytest.fail(f"Expected odds column '{col}' not found in DataFrame.")

    def test_mixed_delimiter_handling(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        # Find the row that originally had 'E0\t05/10/21,15:00,Wolves...'
        problematic_row = df[(df['HomeTeam'] == 'Wolves') & (df['AwayTeam'] == 'Southampton')]
        assert not problematic_row.empty, "Row with mixed delimiter entry was dropped or not found."
        # Verify that 'Div' and 'match_date' for this row are correctly parsed as separate fields
        assert problematic_row['Div'].iloc[0] == 'E0', "'Div' column from mixed delimiter row not handled correctly."
        assert problematic_row['match_date'].iloc[0] == datetime(2021, 10, 5, 15, 0), "'match_date' column from mixed delimiter row not handled correctly."

    def test_get_market_odds_columns(self, mock_data, mock_config):
        df = self.full_ingestion_process(mock_data, mock_config)
        preprocessor = Preprocessor(mock_config)

        # Test MATCH_ODDS
        match_odds_cols = preprocessor.get_market_odds_columns(df, Market.MATCH_ODDS)
        assert 'B365' in match_odds_cols
        assert 'B365H' in match_odds_cols['B365']
        assert 'Max' in match_odds_cols
        assert 'MaxH' in match_odds_cols['Max']
        
        # Test OVER_UNDER_2_5
        over_under_odds_cols = preprocessor.get_market_odds_columns(df, Market.OVER_UNDER_2_5)
        assert 'B365' in over_under_odds_cols
        assert 'B365>2.5' in over_under_odds_cols['B365']
        assert 'P' in over_under_odds_cols
        assert 'P>2.5' in over_under_odds_cols['P']

        # Test ASIAN_HANDICAP
        asian_handicap_odds_cols = preprocessor.get_market_odds_columns(df, Market.ASIAN_HANDICAP)
        assert 'AHh' in asian_handicap_odds_cols
        assert 'AHh' in asian_handicap_odds_cols['AHh'] # AHh is a special case with no provider prefix
        assert 'B365' in asian_handicap_odds_cols
        assert 'B365AHH' in asian_handicap_odds_cols['B365']
        assert 'AvgCAHH' in asian_handicap_odds_cols['Avg'] # Check a closing odd

        # Test an empty DataFrame to ensure ValueError is raised
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="No odds columns found for market type: MATCH_ODDS in the DataFrame."):
            preprocessor.get_market_odds_columns(empty_df, Market.MATCH_ODDS)

        # Test a DataFrame with no matching odds columns (even if not empty)
        df_no_odds = pd.DataFrame({'Div': ['E0'], 'HomeTeam': ['TeamA']})
        with pytest.raises(ValueError, match="No odds columns found for market type: MATCH_ODDS in the DataFrame."):
            preprocessor.get_market_odds_columns(df_no_odds, Market.MATCH_ODDS)
