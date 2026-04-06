"""Signal Mining Machine: discover alpha signals from CRSP/Compustat data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.mining.config import MiningConfig
from src.mining.runner import run_mining


def main():
    pipeline_config = PipelineConfig.from_yaml("config/pipeline.yaml")
    mining_config = MiningConfig.from_yaml("config/mining.yaml")
    run_mining(pipeline_config, mining_config)


if __name__ == "__main__":
    main()
