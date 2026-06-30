from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = RESULTS_DIR / "figures"

REPORTS_DIR = ROOT_DIR / "reports"

RAW_FILE = RAW_DIR / "Base Slim Morte cancer de utero.xlsx"
INTERIM_FILE = INTERIM_DIR / "base_limpa.csv"

TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

POPULATION_SIZE = 30
N_GENERATIONS = 50
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.05
ELITISM_SIZE = 2

NN_EPOCHS = 50
NN_BATCH_SIZE = 16
