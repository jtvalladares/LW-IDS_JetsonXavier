from params import *
from JetsonServer import LightCardServer


def main():

    # Initialize the server
    STORAGE_FILE = "/results/LC_times_E{}.{}_C{}.csv"
    STORAGE_FILE = STORAGE_FILE.format(NUM_EXP, NUM_TEST, NUM_CLASS)
    server = LightCardServer(IP_HOST, PORT_NUM, STORAGE_FILE)
    # Start the server
    server.start()

    # Load model
    model_path = "models/model_F{}.{}__results.pkl_class{}.pkl"
    server.load_model(model_path.format(NUM_EXP, NUM_TEST, NUM_CLASS))

    # Handle client connections
    server.handle_client()

    # Close the server
    server.close()


if __name__ == "__main__":
    main()

