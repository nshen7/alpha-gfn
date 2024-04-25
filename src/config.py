import torch
import pandas as pd

# Fixed hyperparameters
n_hid_units = 512
n_episodes = 10_000
learning_rate = 3e-3

WINDOW_SIZE = 5 # FUTURE TODO: include window size in action space
MAX_EXPR_LENGTH = 20 # Maximum number of tokens in sequence
RESCALE_FACTOR = 100 # Rescaling factor for reward computation
DEVICE = torch.device('cpu')

## Action Space
# Begin
BEG = ["BEG"] # encoded as 0
# Operator names
UNARY = ["ops_abs", "ops_log", "ops_roll_std"] # encoded as 1-3
BINARY = ["ops_add", "ops_subtract", "ops_multiply", "ops_divide", "ops_roll_corr"] # encoded as 4-8
# Features
FEATURES = ["$open", "$close", "$high", "$low", "$volume"] # encoded as 9-13
# End
SEP = ["SEP"] # encoded as 14

# Complete action space
ACTION_SPACE = BEG + UNARY + BINARY + FEATURES + SEP

## Size of action subspace
SIZE_BEG = len(BEG)
SIZE_UNARY = len(UNARY)
SIZE_BINARY = len(BINARY)
SIZE_FEATURE = len(FEATURES)
SIZE_SEP = len(SEP)

SIZE_ACTION = SIZE_BEG + SIZE_UNARY + SIZE_BINARY + SIZE_FEATURE + SIZE_SEP # = 14

# Start indices of each action subset
OFFSET_BEG = 0
OFFSET_UNARY = OFFSET_BEG + SIZE_BEG # = 1
OFFSET_BINARY = OFFSET_UNARY + SIZE_UNARY # = 4
OFFSET_FEATURE = OFFSET_BINARY + SIZE_BINARY # = 9
OFFSET_SEP = OFFSET_FEATURE + SIZE_FEATURE # = 14

## Import data
FOWARD_RETURN = pd.read_csv("../data/processed/ForwardReturn.csv", index_col=0)
OPEN = pd.read_csv("../data/processed/Open.csv", index_col=0)
CLOSE = pd.read_csv("../data/processed/Close.csv", index_col=0)
HIGH = pd.read_csv("../data/processed/High.csv", index_col=0)
LOW = pd.read_csv("../data/processed/Low.csv", index_col=0)
VOLUME = pd.read_csv("../data/processed/Volume.csv", index_col=0)
FEATURE_DATA = {"$open": OPEN, "$close": CLOSE, "$high": HIGH, "$low": LOW, "$volume": VOLUME}