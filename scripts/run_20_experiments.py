import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == "__main__":
    for experiment_number in range(1, 21):
        run_experiment(experiment_number=experiment_number)
