# Snake AI

This project explores two different approaches to building an AI to play the classic game of Snake:

1.  **Neural Network Approach:** A supervised learning approach where a neural network is trained on data generated from a human playing the game.
2.  **Reinforcement Learning Approach:** An unsupervised learning approach using Deep Q-Networks (DQN) where the agent learns to play the game through trial and error.

## Installation

To run this project, you need to have Python 3 installed. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Neural Network

To train the neural network, you first need to generate training data by playing the game yourself.

1.  **Generate Data:**
    ```bash
    python neural_network/snake.py
    ```
    Play the game, and the data will be saved to a `.csv` file in the `neural_network/data` directory.

2.  **Train the Model:**
    ```bash
    python neural_network/train.py
    ```
    This will train the neural network on the generated data and save the model.

3.  **Run the AI:**
    ```bash
    python neural_network/run.py
    ```
    This will run the game with the trained neural network playing.

#### Arguments

*   `train.py`:
    *   `model`: The type of model to train.
        *   Choices: `knn`, `svm`, `tf`, `pytorch`
*   `run.py`:
    *   `file`: Path to the trained model file (.pth, .h5, or .pkl).

### Reinforcement Learning

To train the reinforcement learning agent:

```bash
python reinforcement_learning/dqn.py
```

This will train the DQN agent and save the model. You can see the agent learning in real-time.

## Demo

Here is a video of the AI in action:

<video src="snake-video.mov" width="400" controls>
  Your browser does not support the video tag.
</video>
