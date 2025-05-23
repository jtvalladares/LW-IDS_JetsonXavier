from params import *
from JetsonClient import JetsonNanoClient


def main():
    # Initialize the server
    STORAGE_FILE = "/results/times_E{}.{}_C{}.csv"
    STORAGE_FILE = STORAGE_FILE.format(NUM_EXP, NUM_TEST, NUM_CLASS)
    client = JetsonNanoClient(IP_HOST, PORT, STORAGE_FILE, NUM_REGS)
    # Connect to the server
    client.connect()

    # Load test files
    test_path = "test_sets/F{}.{}__results.pkl_class{}_test.npy"
    client.load_test_file(test_path.format(NUM_EXP, NUM_TEST, NUM_CLASS))
    client.tx_rx()
    

if __name__ == "__main__":
    main()