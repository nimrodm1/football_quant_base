from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

class Market(Enum):
    MATCH_ODDS = "MATCH_ODDS"
    OVER_UNDER_2_5 = "OVER_UNDER_2_5"
    ASIAN_HANDICAP = "ASIAN_HANDICAP"

class Outcomes(Enum):
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    OVER_25 = "over"
    UNDER_25 = "under"

@dataclass
class DataConfig:
    RAW_COL_MAP: Dict[str, str] = field(default_factory=lambda: {
        "Div": "Div", "Date": "Date", "Time": "Time", "HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam",
        "FTHG": "FTHG", "FTAG": "FTAG", "FTR": "FTR", "HTHG": "HTHG", "HTAG": "HTAG", "HTR": "HTR",
        "Referee": "Referee", "HS": "HS", "AS": "AS", "HST": "HST", "AST": "AST", "HF": "HF",
        "AF": "AF", "HC": "HC", "AC": "AC", "HY": "HY", "AY": "AY", "HR": "HR", "AR": "AR",
        "B365_H": "B365H", "B365_D": "B365D", "B365_A": "B365A",
        "B_H": "BWH", "B_D": "BWD", "B_A": "BWA",
        "I_H": "IWH", "I_D": "IWD", "I_A": "IWA",
        "PS_H": "PSH", "PS_D": "PSD", "PS_A": "PSA",
        "WH_H": "WHH", "WH_D": "WHD", "WH_A": "WHA",
        "VC_H": "VCH", "VC_D": "VCD", "VC_A": "VCA",
        "Max_H": "MaxH", "Max_D": "MaxD", "Max_A": "MaxA",
        "Avg_H": "AvgH", "Avg_D": "AvgD", "Avg_A": "AvgA",
        "B365_2_5": "B365>2.5", "B365_U2_5": "B365<2.5",
        "P_2_5": "P>2.5", "P_U2_5": "P<2.5",
        "Max_2_5": "Max>2.5", "Max_U2_5": "Max<2.5",
        "Avg_2_5": "Avg>2.5", "Avg_U2_5": "Avg<2.5",
        "AHh": "AHh", "B365AHH": "B365AHH", "B365AHA": "B365AHA",
        "PAHH": "PAHH", "PAHA": "PAHA",
        "MaxAHH": "MaxAHH", "MaxAHA": "MaxAHA",
        "AvgAHH": "AvgAHH", "AvgAHA": "AvgAHA",
        "B365C_H": "B365CH", "B365C_D": "B365CD", "B365C_A": "B365CA",
        "BWC_H": "BWCH", "BWC_D": "BWCD", "BWC_A": "BWCA",
        "IWC_H": "IWCH", "IWC_D": "IWCD", "IWC_A": "IWCA",
        "PSC_H": "PSCH", "PSC_D": "PSCD", "PSC_A": "PSCA",
        "WHC_H": "WHCH", "WHC_D": "WHCD", "WHC_A": "WHCA",
        "VCC_H": "VCCH", "VCC_D": "VCCD", "VCC_A": "VCCA",
        "MaxC_H": "MaxCH", "MaxC_D": "MaxCD", "MaxC_A": "MaxCA",
        "AvgC_H": "AvgCH", "AvgC_D": "AvgCD", "AvgC_A": "AvgCA",
        "B365C_2_5": "B365C>2.5", "B365C_U2_5": "B365C<2.5",
        "PC_2_5": "PC>2.5", "PC_U2_5": "PC<2.5",
        "MaxC_2_5": "MaxC>2.5", "MaxC_U2_5": "MaxC<2.5",
        "AvgC_2_5": "AvgC>2.5", "AvgC_U2_5": "AvgC<2.5",
        "AHCh": "AHCh", "B365CAHH": "B365CAHH", "B365CAHA": "B365CAHA",
        "PCAHH": "PCAHH", "PCAHA": "PCAHA",
        "MaxCAHH": "MaxCAHH", "MaxCAHA": "MaxCAHA",
        "AvgCAHH": "AvgCAHH", "AvgCAHA": "AvgCAHA"
    })
    DATE_FORMATS: List[str] = field(default_factory=lambda: ["%d/%m/%y", "%d/%m/%Y"])
    TIME_FORMAT: str = "%H:%M"
    ODDS_COL_PATTERNS: Dict[Market, Dict[str, Dict[str, str]]] = field(default_factory=lambda: {
        Market.MATCH_ODDS: {
            "pre_match": {
                "B365": "^B365[HDA]$", "BW": "^BW[HDA]$", "IW": "^IW[HDA]$", "PS": "^PS[HDA]$",
                "WH": "^WH[HDA]$", "VC": "^VC[HDA]$", "Max": "^Max[HDA]$", "Avg": "^Avg[HDA]$"
            },
            "closing": {
                "B365": "^B365C[HDA]$", "BW": "^BWC[HDA]$", "IW": "^IWC[HDA]$", "PS": "^PSC[HDA]$",
                "WH": "^WHC[HDA]$", "VC": "^VCC[HDA]$", "Max": "^MaxC[HDA]$", "Avg": "^AvgC[HDA]$"
            }
        },
        Market.OVER_UNDER_2_5: {
            "pre_match": {
                "B365": "^B365[<>]2.5$", "P": "^P[<>]2.5$", "Max": "^Max[<>]2.5$", "Avg": "^Avg[<>]2.5$"
            },
            "closing": {
                "B365": "^B365C[<>]2.5$", "P": "^PC[<>]2.5$", "Max": "^MaxC[<>]2.5$", "Avg": "^AvgC[<>]2.5$"
            }
        },
        Market.ASIAN_HANDICAP: {
            "pre_match": {
                "AHh": "^AHh$", "B365": "^B365AH[HA]$", "P": "^PAH[HA]$", "Max": "^MaxAH[HA]$", "Avg": "^AvgAH[HA]$"
            },
            "closing": {
                "AHCh": "^AHCh$", "B365": "^B365CAH[HA]$", "P": "^PCAH[HA]$", "Max": "^MaxCAH[HA]$", "Avg": "^AvgCAH[HA]$"
            }
        }
    })
    MISSING_VALUE_PLACEHOLDERS: List[Any] = field(default_factory=lambda: ["", "#VALUE!", "-", "None", "N/A", 'NA']) 
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    TEAM_NAME_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "Man United": "Man Utd", "Arsenal FC": "Arsenal", "Leicester": "Leicester City"
    })
    SCHEMA_DATA_TYPES: Dict[str, str] = field(default_factory=lambda: {
        "Div": "string", "Date": "string", "Time": "string", "HomeTeam": "string", "AwayTeam": "string",
        "FTHG": "Int64", "FTAG": "Int64", "FTR": "string", "HTHG": "Int64", "HTAG": "Int64", "HTR": "string",
        "Referee": "string", "HS": "Int64", "AS": "Int64", "HST": "Int64", "AST": "Int64", "HF": "Int64",
        "AF": "Int64", "HC": "Int64", "AC": "Int64", "HY": "Int64", "AY": "Int64", "HR": "Int64",
        "AR": "Int64",
        "B365H": "float64", "B365D": "float64", "B365A": "float64",
        "BWH": "float64", "BWD": "float64", "BWA": "float64",
        "IWH": "float64", "IWD": "float64", "IWA": "float64",
        "PSH": "float64", "PSD": "float64", "PSA": "float64",
        "WHH": "float64", "WHD": "float64", "WHA": "float64",
        "VCH": "float64", "VCD": "float64", "VCA": "float64",
        "MaxH": "float64", "MaxD": "float64", "MaxA": "float64",
        "AvgH": "float64", "AvgD": "float64", "AvgA": "float64",
        "B365>2.5": "float64", "B365<2.5": "float64",
        "P>2.5": "float64", "P<2.5": "float64",
        "Max>2.5": "float64", "Max<2.5": "float64",
        "Avg>2.5": "float64", "Avg<2.5": "float64",
        "AHh": "float64", "B365AHH": "float64", "B365AHA": "float64",
        "PAHH": "float64", "PAHA": "float64",
        "MaxAHH": "float64", "MaxAHA": "float64",
        "AvgAHH": "float64", "AvgAHA": "float64",
        "B365CH": "float64", "B365CD": "float64", "B365CA": "float64",
        "BWCH": "float64", "BWCD": "float64", "BWCA": "float64",
        "IWCH": "float64", "IWCD": "float64", "IWCA": "float64",
        "PSCH": "float64", "PSCD": "float64", "PSCA": "float64",
        "WHCH": "float64", "WHCD": "float64", "WHCA": "float64",
        "VCCH": "float64", "VCCD": "float64", "VCCA": "float64",
        "MaxCH": "float64", "MaxCD": "float64", "MaxCA": "float64",
        "AvgCH": "float64", "AvgCD": "float64", "AvgCA": "float64",
        "B365C>2.5": "float64", "B365C<2.5": "float64",
        "PC>2.5": "float64", "PC<2.5": "float64",
        "MaxC>2.5": "float64", "MaxC<2.5": "float64",
        "AvgC>2.5": "float64", "AvgC<2.5": "float64",
        "AHCh": "float64", "B365CAHH": "float64", "B365CAHA": "float64",
        "PCAHH": "float64", "PCAHA": "float64",
        "MaxCAHH": "float64", "MaxCAHA": "float64",
        "AvgCAHH": "float64", "AvgCAHA": "float64"
    })

@dataclass
class FeatureConfig(DataConfig):
    feature_list: List[str] = field(default_factory=list)
    time_decay_scaling_factor: float = 0.0
    reference_date_for_decay: str = ""
    implied_prob_bookmakers_match_odds: List[str] = field(default_factory=lambda: ["B365", "PS", "Avg"])
    implied_prob_bookmakers_over_under: List[str] = field(default_factory=lambda: ["B365", "P", "Avg"])
    closing_odds_preferred_provider: Optional[str] = "PS"

@dataclass
class ModellingConfig(FeatureConfig):
    sampling: Dict[str, Any] = field(default_factory=lambda: {
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.95,
        "random_seed": 42
    })
    priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "mu": {"mean": 0.0, "sd": 1.0},
        "h_adv": {"mean": 0.3, "sd": 0.2},
        "sigma_att": {"sd": 0.5},
        "sigma_def": {"sd": 0.5},
        "alpha": {"sd": 0.1}
    })
    prediction: Dict[str, Any] = field(default_factory=lambda: {
        "max_goals": 10,
        "n_samples": 2000
    })

@dataclass
class StrategyConfig(ModellingConfig):
    name: str = "kelly_optimal"
    value_bet_threshold: float = 0.02
    max_match_exposure: float = 0.05
    kelly_fraction_k: float = 0.25
    flat_stake_unit: float = 10.0
    markets_to_monitor: List[str] = field(default_factory=lambda: ["MATCH_ODDS", "OVER_UNDER_2_5"])

@dataclass
class BacktestConfig(StrategyConfig):
    initial_bankroll: float = 1000.0
    training_window_months: int = 24
    min_training_data_points: int = 100
    retrain_frequency: int = 7
    default_odds_provider_pre_match: str = "Avg"
    default_odds_provider_close: str = "PS"
    eval_metrics: List[str] = field(default_factory=lambda: ["roi", "pnl", "brier_score", "log_loss", "clv_score"])
