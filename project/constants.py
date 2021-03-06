import time

# params

# quantum circuit
N_WIRES = 4
N_LAYERS = 3

ENV_NAME = "Blackjack-v0"
BUFFER_SIZE = 10000
TOTAL_TIMESTEPS = 20000
LEARNING_STARTS = 10000
TRAIN_FREQ = 1
BATCH_SIZE = 32
TARGET_NET_UPDATE_FREQ = 500 # net_target = net with that frequency

LR = 0.5
GAMMA = 0.98

# epsilon params
START_EPSILON = 0.8
END_EPSILON = 0.01
EXPLORATION_FRACTION = 0.75 # percentage of the timesteps exploring

SEED = 13 # int(time.time())
TORCH_DETERMINISTIC = True
CAPTURE_VIDEO = False