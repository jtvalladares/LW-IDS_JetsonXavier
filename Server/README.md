# Model Prediction Docker Runner

This is an edge AI application designed to perform real-time inference over a socket-based internet connection. It connects to a **hosted client** that sends feature data, and the **edge device**—running inside a Docker container—predicts whether the incoming data represents a cyberattack or not using pre-trained models.

The project leverages Python sockets for bidirectional communication and is intended for deployment scenarios where model inference happens remotely (e.g., on a Jetson Nano) based on streamed data.

## 📁 Model Naming Convention

Pre-trained models are located in the `src/models` directory and follow this naming format:

```python
"model_F{NUM_EXP}.{NUM_TEST}__results.pkl_class{NUM_CLASS}.pkl"
```


## 🚀 Running the Container

You **do not need to modify any `.env` files**. Just pass environment variables directly in the terminal when starting the container using Docker Compose.

### ✅ Required Environment Variables

| Variable         | Description                           | Fixed or User-Defined |
|------------------|---------------------------------------|------------------------|
| `IP_HOST`        | IP address of the host device         | Fixed: `192.168.0.126` |
| `PORT_NUM`       | Port used for socket communication    | Fixed: `7632`          |
| `NUM_MODELS`     | Total number of models to iterate     | Fixed: `4`             |
| `NUM_CLASSES`    | Total number of classification classes| Fixed: `3`             |
| `NUM_ROWS_IN_TEST` | Rows in test dataset                | Fixed: `117177`        |
| `LOCAL_TEST`     | Whether to use local test mode (0/1)  | Fixed: `0`             |
| `NUM_EXP`        | Experiment number                     | 🔁 User-provided       |
| `NUM_TEST`       | Test number                           | 🔁 User-provided       |
| `NUM_CLASS`      | Class number                          | 🔁 User-provided       |

## 🔧 Example Usage

To run the container on the Jetson Nano (edge device), follow these steps:

1. **SSH into the Jetson Nano**:
    ```bash
    ssh jetson10@IP_HOST
    ```

2. **Navigate to your project directory**, then enter the `JetsonNano` folder:
    ```bash
    cd path/to/LightCard_Jetson_Nano/JetsonNano
    ```

3. **If created, deleted results folder for new data logging**: 
     ```bash
    sudo rm -rf results
    ```

3. **Run the container using Docker Compose**, specifying the environment variables for the model:
    ```bash
    NUM_EXP=1 NUM_TEST=1 NUM_CLASS=1 \
    IP_HOST=192.168.0.126 PORT_NUM=7632 \
    NUM_MODELS=4 NUM_CLASSES=3 NUM_ROWS_IN_TEST=117177 \
    LOCAL_TEST=0 docker compose up --build
    ```
