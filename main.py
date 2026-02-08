import pandas as pd
from quant_football.core.config import BacktestConfig
from quant_football.backtesting.backtester import Backtester
from quant_football.backtesting.pipelines import DataPipeline, ModelPipeline
from quant_football.modelling import BayesianPoissonGLMM 
from quant_football.strategy import KellyOptimalStrategy

def run_pipeline(csv_paths: list[str]):
    # 1. Configuration
    config = BacktestConfig(
        initial_bankroll=1000.0,
        retrain_frequency=14,
        training_window_months=24,
        min_training_data_points=380,
        default_odds_provider_pre_match="Avg",
        default_odds_provider_close="PS"
    )

    # 2. Performance & Sampler Settings
    sampler_settings = {
        "nuts_sampler": "nutpie",
        "mode": "NUMBA"
    }

    # 3. Initialize Components
    model = BayesianPoissonGLMM(config={
        'sampling': {'draws': 1000, 'tune': 500, 'chains': 2, 'target_accept': 0.9}
    }) 
    
    strategy = KellyOptimalStrategy(kelly_fraction=0.05)
    data_pipeline = DataPipeline(config)
    model_pipeline = ModelPipeline(model, sampler_config=sampler_settings)
    
    backtester = Backtester(
        config=config,
        data_pipeline=data_pipeline,
        model_pipeline=model_pipeline,
        strategy=strategy
    )

    # 4. Run the Engine
    print(f"⏳ Loading {len(csv_paths)} files and starting Backtest...")
    print(f"🚀 Sampler: {sampler_settings['nuts_sampler']} | Mode: {sampler_settings['mode']}")
    
    # We pass the list of strings; the pipeline handles the reading
    results = backtester.run_backtest(csv_paths)

    # 5. Output Performance
    print("\n" + "="*30)
    print("🏆 BACKTEST RESULTS 🏆")
    print("="*30)
    if results.get('status') == "no_bets_placed":
        print("Outcome: No bets were placed. Check data coverage.")
    else:
        print(f"Final Bankroll:  £{results.get('final_bankroll', 0.0):.2f}")
        print(f"Total PnL:       £{results.get('pnl', 0.0):.2f}")
        print(f"ROI:             {results.get('roi', 0.0):.2%}")
        print(f"Total Bets:      {results.get('count', 0)}")
        print(f"clv_score:        {results.get('clv_score', 0.0):.4f}")
        print(f"Brier Score:      {results.get('brier_score', 0.0):.4f}")
        print(f"Log Loss:        {results.get('log_loss', 0.0):.4f}")
    print("="*30)

    if backtester.history:
        pd.DataFrame(backtester.history).to_csv("backtest_log.csv", index=False)
        print("💾 Detailed logs saved to 'backtest_log.csv'")

if __name__ == "__main__":
    run_pipeline(["data/21_22.csv", "data/22_23.csv"])