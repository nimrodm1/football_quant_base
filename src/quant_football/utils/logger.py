import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "quant_football.log")
    ]
)

# This is the variable the rest of your app is looking for
logger = logging.getLogger("quant_football")