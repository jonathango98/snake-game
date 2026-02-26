import argparse
import pandas as pd
import glob
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras import layers, models

# Assuming your class and list are in nn.py
from nn import SnakeNet, FEATURE_COLS

def load_and_balance(file_pattern, seed=42):
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found for {file_pattern}")
    
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    turns = df[df['action'] != 0]
    straight = df[df['action'] == 0]
    
    # Matching the balancing logic from your neural network setup
    n = min(len(straight), len(turns))
    df_balanced = pd.concat([turns, straight.sample(n=n, random_state=seed)])
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    X = df_balanced[FEATURE_COLS]
    y = df_balanced['action']
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def main():
    parser = argparse.ArgumentParser(description="Train Snake AI models.")
    parser.add_argument("model", choices=["knn", "svm", "tf", "pytorch"], 
                        default="rf", help="Choose model: rf, knn, svm, nb, tf, pytorch")
    args = parser.parse_args()

    X_train_df, X_test_df, y_train_df, y_test_df = load_and_balance("data/snake_data_*.csv")
    model_type = args.model

    if model_type == "pytorch":
        X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
        y_train = torch.tensor(y_train_df.values, dtype=torch.long)
        
        model = SnakeNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)

        print(f"Training PyTorch SnakeNet on {len(X_train)} samples...")
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), "model/snake_model_pytorch.pth")
        print("Saved: snake_model_pytorch.pth")

    elif model_type == "tf":
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(17,)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_df, y_train_df, epochs=50, verbose=1)
        model.save("model/snake_model_tf.h5")
        print("Saved: snake_model_tf.h5")

    else:
        clfs = {
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(probability=True),
        }
        model = clfs[model_type]
        model.fit(X_train_df, y_train_df)
        joblib.dump(model, f"model/snake_model_{model_type}.pkl")
        print(f"Saved: snake_model_{model_type}.pkl")

if __name__ == "__main__":
    main()