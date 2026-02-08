# Football Quant: Bayesian Poisson Backtesting

A framework for football match forecasting and historical backtesting using **Bayesian Poisson GLMMs**. 

This repository covers the foundational infrastructure for data ingestion, model fitting, and betting strategy evaluation.

## Development Workflow
The package was built using a **multi-agent workflow** based on https://github.com/nimrodm1/football_quant_dev_orchestration. 

* **Reusable Workflow:** I developed a single, modular agentic workflow that was reused across five distinct development sprints: Data, Features, Model, Strategy, and Backtesting. 
* **Prompt-Driven Context:** The underlying orchestration logic remained unchanged throughout the project; the transition between sprints was achieved solely by updating the system prompts to specify the agents' roles with sprint-specific details.
* **Manual Refinement:** While the system scaffolded the architecture and boilerplate (roughly 80% of the codebase), I finalised the implementation to handle specific technical requirements and the connectivity between the different modules. 

## Technical Architecture

### 1. Data & Preprocessing
Tailored for **football-data.co.uk** structures:
* **Standardisation:** Header mapping and team name consistency across seasons.
* **Deterministic IDs:** Unique match and team indexing for categorical compatibility with the Bayesian backend.
* **Validation:** Explicit schema enforcement using `Int64` and `StringDtype` to ensure stable sampling.

### 2. Feature Engineering
* **Market Analysis:** Logic to calculate implied probabilities and overround from decimal odds to filter data errors.
* **Odds Processing:** Implements a selection layer that prioritises specific bookmakers with automated fallback to market maximums provided in the dataset.

### 3. Modelling & Strategy
* **Inference:** Poisson GLMM implemented in `PyMC` using the Rust-based **Nutpie** sampler for faster convergence.
* **Betting Logic:** An abstract `BaseStrategy` class that links match predictions to market odds via standardised `MatchPrediction` and `MatchOdds` objects.
* **Staking:** Includes a **Kelly Criterion** implementation for position sizing, with configurable Expected Value (EV) and exposure constraints.

### 4. Backtesting
The pipeline uses an expanding-window approach to simulate historical performance. It retrains the model as new data becomes available to measure predictive accuracy and strategy drawdown in a realistic environment.

## Tech Stack
* **Probabilistic Programming:** PyMC & Nutpie (Rust-based sampler for faster MCMC sampling)
* **Orchestration:** LangGraph (Multi-agent workflow)
* **Data Science:** Pandas, NumPy
* **Environment:** Conda / Miniforge

## Note on Project Status
This repository contains the core backtesting and modelling infrastructure. Further strategy refinements and execution modules are maintained in a private repository.