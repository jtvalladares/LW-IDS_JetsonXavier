import os 

IP_HOST = os.getenv("IP_HOST", "10.8.33.129")
PORT_NUM = int(os.getenv("PORT_NAME", 7632))

NUM_MODELS = int(os.getenv("NUM_MODELS", 4))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 3))

NUM_ROWS_IN_TEST = int(os.getenv("NUM_ROWS_IN_TEST", 117177))

LOCAL_TEST = int(os.getenv("LOCAL_TEST", 0))

NUM_EXP = int(os.getenv("NUM_EXP", 1))
NUM_TEST = int(os.getenv("NUM_TEST", 1))
NUM_CLASS = int(os.getenv("NUM_CLASS", 1))
