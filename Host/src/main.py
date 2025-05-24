from params import *
from JetsonClient import JetsonXavierClient
import itertools


def main():
    exp_list = list(range(1, 4))
    test_list = list(range(1, 5))
    class_list = list(range(1, 4))

    comb = list(itertools.product(exp_list, test_list, class_list))

    for NUM_EXP,NUM_TEST,NUM_CLASS in comb:
        print(f"Starting with file E{NUM_EXP}.{NUM_TEST}_C{NUM_CLASS}.csv")
        # Initialize the server
        STORAGE_FILE = "/results/times_E{}.{}_C{}.csv"
        STORAGE_FILE = STORAGE_FILE.format(NUM_EXP, NUM_TEST, NUM_CLASS)
        client = JetsonXavierClient(IP_HOST, PORT, STORAGE_FILE, NUM_REGS)
        # Connect to the server
        client.connect()

        # Load test files
        test_path = "test_sets/E{}.{}__results.pkl_class{}_test.npy"
        client.load_test_file(test_path.format(NUM_EXP, NUM_TEST, NUM_CLASS))
        client.tx_rx()
    

if __name__ == "__main__":
    main()