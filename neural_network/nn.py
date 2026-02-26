import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os

FEATURE_COLS = [
    "col_left", "col_straight", "col_right",
    "hit_left", "hit_straight", "hit_right",
    "cur_left", "cur_up", "cur_right", "cur_down",
    "food_left", "food_up", "food_right", "food_down",
    "dist_to_tail", "body_left_ratio", "body_right_ratio"
]
LABEL_COL = "action"


class SnakeNet(nn.Module):
    def __init__(self, in_dim=len(FEATURE_COLS), hidden=256, out_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(17, 512), # Or higher if you add more LiDAR rays
            nn.ReLU(),
            nn.Dropout(0.2),    # Prevents overfitting to specific game patterns
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3) 
        )

    def forward(self, x):
        return self.network(x)