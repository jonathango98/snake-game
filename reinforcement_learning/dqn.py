import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from collections import deque
from snake import SnakeGameAI, Direction, Point

# --- CONFIGURATION ---
NUM_GAMES = 10           # Number of parallel games
BATCH_SIZE = 1000        # Training batch size
MAX_MEMORY = 100_000     # Replay memory size
LR = 0.001               # Learning Rate
CHECKPOINT_PATH = './model/parallel_checkpoint.pth'

# M1/M2 Optimization
# NOTE: If you still get 'trace trap' crashes, change this to 'cpu'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {DEVICE}")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save_checkpoint(self, n_games, optimizer, file_name=CHECKPOINT_PATH):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'n_games': n_games
        }, file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert lists to tensors and move to GPU
        states = torch.tensor(np.array(states), dtype=torch.float).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(DEVICE)

        # Predict Q values for the whole batch
        pred = self.model(states)

        target = pred.clone()
        for i in range(len(dones)):
            Q_new = rewards[i]
            if not dones[i]:
                # Bellman Equation
                Q_new = rewards[i] + self.gamma * torch.max(self.model(next_states[i]))
            
            target[i][torch.argmax(actions[i]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class ParallelAgent:
    def __init__(self):
        self.n_games = 0 # Total games completed across all threads
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        self.model = Linear_QNet(13, 256, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # --- RESUME LOGIC ---
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Loading checkpoint: {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.model.load_state_dict(checkpoint['model_state'])
            if 'optimizer_state' in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.n_games = checkpoint['n_games']
            self.model.eval()
            print(f"Resumed! Total Games Played: {self.n_games}")

    def get_batch_action(self, states):
        # Decaying Epsilon
        self.epsilon = 80 - (self.n_games // NUM_GAMES) # Adjust decay speed
        final_moves = []
        
        # 1. Random Moves (Exploration)
        if random.randint(0, 200) < self.epsilon:
            for _ in range(len(states)):
                move = [0,0,0]
                move[random.randint(0, 2)] = 1
                final_moves.append(move)
                
        # 2. AI Moves (Exploitation) - ONE PASS on M1 GPU
        else:
            states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(DEVICE)
            predictions = self.model(states_tensor) # (50, 3)
            
            # Convert GPU tensor back to CPU list of moves
            max_idxs = torch.argmax(predictions, dim=1).tolist()
            for idx in max_idxs:
                move = [0,0,0]
                move[idx] = 1
                final_moves.append(move)
                
        return final_moves

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

def train_parallel():
    agent = ParallelAgent()
    
    # Initialize 50 Games
    # Game 0: Render=True (Viewer)
    # Game 1-49: Render=False (Headless)
    print(f"Initializing {NUM_GAMES} parallel environments...")
    games = [SnakeGameAI(render_mode=(i==0)) for i in range(NUM_GAMES)]
    
    record = 0
    print("Training Started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # 1. Get States from ALL 50 games
            states_old = [game.get_state() for game in games]

            # 2. Get Actions for ALL 50 games (Vectorized)
            final_moves = agent.get_batch_action(states_old)

            # 3. Step ALL 50 games
            # We use a list to store results for this frame
            step_results = []
            for i, game in enumerate(games):
                reward, done, score = game.play_step(final_moves[i])
                step_results.append((reward, done, score))

            # 4. Get New States
            states_new = [game.get_state() for game in games]
            
            # 5. Process Batch Results
            rewards, dones, scores = zip(*step_results)
            
            # Train on this batch immediately (Short Memory)
            agent.train_short_memory(states_old, final_moves, rewards, states_new, dones)

            # Add to Replay Memory
            for i in range(NUM_GAMES):
                agent.remember(states_old[i], final_moves[i], rewards[i], states_new[i], dones[i])
                
                # Handling Game Over
                if dones[i]:
                    games[i].reset()
                    agent.n_games += 1
                    
                    # Log progress (Only for the Viewer game to avoid console spam)
                    if i == 0:
                        if scores[i] > record:
                            record = scores[i]
                            agent.model.save_checkpoint(agent.n_games, agent.trainer.optimizer)
                        
                        print(f'Game {agent.n_games} | Score {scores[i]} | Record {record}')
                        
                        # Train Long Memory whenever the "Viewer" game finishes
                        agent.train_long_memory()

    except KeyboardInterrupt:
        print("\nParallel Training Interrupted! Saving...")
        agent.model.save_checkpoint(agent.n_games, agent.trainer.optimizer)
        print("Saved.")

if __name__ == '__main__':
    train_parallel()