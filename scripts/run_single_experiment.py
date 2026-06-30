import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == "__main__":
    run_experiment(experiment_number=1)
