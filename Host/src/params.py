import os

IP_HOST = os.getenv("IP_HOST", "10.8.33.137")
PORT = int(os.getenv("PORT", 7631))

NUM_EXP = int(os.getenv("NUM_EXP", 1))
NUM_TEST = int(os.getenv("NUM_TEST", 2))
NUM_CLASS = int(os.getenv("NUM_CLASS", 1))

NUM_REGS = int(os.getenv("NUM_REGS", 117177))