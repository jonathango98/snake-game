import argparse
import pygame
import joblib
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from snake import SnakeEngine, SnakeUI
from nn import SnakeNet, FEATURE_COLS

def run_ai(model_path):
    # Extension-based detection
    if model_path.endswith(".pth"):
        model = SnakeNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        lib = "pytorch"
    elif model_path.endswith(".h5"):
        model = tf.keras.models.load_model(model_path)
        lib = "tf"
    else:
        model = joblib.load(model_path)
        lib = "skl"

    engine = SnakeEngine(log_data=False)
    ui = SnakeUI(engine)
    tps = 60
    dt = 0.0

    while True:
        engine._pump_input()
        dt += ui.clock.tick(60) / 1000.0

        while dt >= 1.0 / tps:
            dt -= 1.0 / tps
            state = engine.get_state()
            state_df = pd.DataFrame([state], columns=FEATURE_COLS)

            if lib == "pytorch":
                with torch.no_grad():
                    tensor_state = torch.tensor(state_df.values, dtype=torch.float32)
                    output = model(tensor_state)
                    action = torch.argmax(output, dim=1).item()
            elif lib == "tf":
                output = model.predict(state_df, verbose=0)
                action = np.argmax(output[0])
            else:
                action = model.predict(state_df)[0]

            # Execute action
            dx, dy = engine.direction
            if action == 1: intent = (-dy, dx)
            elif action == 2: intent = (dy, -dx)
            else: intent = (dx, dy)

            if engine.step(intent):
                return  # Game over, exit loop
        ui.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Snake AI.")
    parser.add_argument("file", help="Path to model file (.pth, .h5, or .pkl)")
    args = parser.parse_args()
    run_ai(args.file)