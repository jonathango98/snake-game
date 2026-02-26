import pygame
import random
import sys
import csv
from datetime import datetime
import os

# --- THE MASTER ENGINE ---
class SnakeEngine:
    def __init__(self, grid_w=20, grid_h=15, log_data=True):
        self.grid_w = grid_w
        self.grid_h = grid_h

        # Logging toggle
        self.log_data_enabled = log_data
        self.filename = None
        self.csv = None
        self.writer = None

        if self.log_data_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"data/snake_data_{timestamp}.csv"
            self.csv = open(self.filename, "w", newline="")
            self.writer = csv.writer(self.csv)
            self.writer.writerow([
                # Distances (3)
                "col_left", "col_straight", "col_right",

                # NEW: immediate neighbor collision flags (3)
                "hit_left", "hit_straight", "hit_right",

                # Direction one-hot (4)
                "cur_left", "cur_up", "cur_right", "cur_down",

                # Food relative (4)
                "food_left", "food_up", "food_right", "food_down",

                # NEW: tail distance + body balance (2)
                "dist_to_tail", "body_left_ratio", "body_right_ratio",

                # Label
                "action"
            ])

        self.reset()

    def reset(self):
        self.head = (self.grid_w // 2, self.grid_h // 2)
        self.snake = [
            self.head,
            (self.head[0] - 1, self.head[1]),
            (self.head[0] - 2, self.head[1]),
        ]
        self.direction = (1, 0)      # start moving right
        self.pending_dir = None      # buffered key input for next tick
        self.score = 0
        self.food = self._place_food()

    def close(self, delete_file=False):
        """Safely close CSV. Optionally delete the CSV file."""
        if not self.log_data_enabled:
            return

        try:
            if self.csv and not self.csv.closed:
                self.csv.close()
        except Exception:
            pass

        if delete_file and self.filename and os.path.exists(self.filename):
            try:
                os.remove(self.filename)
            except Exception:
                pass

    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if food not in self.snake:
                return food

    def _pump_input(self):
        """Pump events frequently and buffer the latest requested direction."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.pending_dir = (0, -1)
                elif event.key == pygame.K_s:
                    self.pending_dir = (0, 1)
                elif event.key == pygame.K_a:
                    self.pending_dir = (-1, 0)
                elif event.key == pygame.K_d:
                    self.pending_dir = (1, 0)

    def _if_collision(self, pos, will_eat=False):
        """Collision with walls or body.
        Allow moving into the tail cell if the tail will move (i.e., not eating).
        """
        x, y = pos
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return True

        body = self.snake if will_eat else self.snake[:-1]
        return pos in body

    def get_distance_to_collision(self, dir_x, dir_y, max_scan=10):
        """Returns normalized distance (0 to 1) to the nearest collision."""
        for dist in range(1, max_scan + 1):
            scan_pos = (self.head[0] + dir_x * dist, self.head[1] + dir_y * dist)
            if self._if_collision(scan_pos, will_eat=False):
                return dist / max_scan
        return 1.0

    def get_state(self):
        head_x, head_y = self.head
        food_x, food_y = self.food
        dx, dy = self.direction
        tail_x, tail_y = self.snake[-1]

        # Relative directions based on current heading
        left_dir = (-dy, dx)
        straight_dir = (dx, dy)
        right_dir = (dy, -dx)

        # NEW: immediate neighbor collision flags (1 cell away)
        left_cell = (head_x + left_dir[0], head_y + left_dir[1])
        straight_cell = (head_x + straight_dir[0], head_y + straight_dir[1])
        right_cell = (head_x + right_dir[0], head_y + right_dir[1])

        hit_left = int(self._if_collision(left_cell, will_eat=False))
        hit_straight = int(self._if_collision(straight_cell, will_eat=False))
        hit_right = int(self._if_collision(right_cell, will_eat=False))

        # Inside get_state()
        # 1. Distance to Tail (Manhattan Distance)
        dist_to_tail = (abs(tail_x - head_x) + abs(tail_y - head_y)) / (self.grid_w + self.grid_h)

        # 2. Side-to-Side Body Count
        # Check if there is more of 'me' on my left or my right
        body_left = sum(1 for seg in self.snake if (seg[0]-head_x)*dy - (seg[1]-head_y)*dx > 0)
        body_right = sum(1 for seg in self.snake if (seg[0]-head_x)*dy - (seg[1]-head_y)*dx < 0)

        return [
            # Distances (relative lidar)
            self.get_distance_to_collision(left_dir[0], left_dir[1]),
            self.get_distance_to_collision(straight_dir[0], straight_dir[1]),
            self.get_distance_to_collision(right_dir[0], right_dir[1]),

            # NEW: immediate collisions
            hit_left, hit_straight, hit_right,

            # Current direction one-hot (absolute)
            int(dx == -1), int(dy == -1), int(dx == 1), int(dy == 1),

            # Food relative (binary)
            int(food_x < head_x), int(food_y < head_y),
            int(food_x > head_x), int(food_y > head_y),

            dist_to_tail,
            body_left/len(self.snake),
            body_right/len(self.snake)
        ]

    def log_data(self, intent):
        """Log (state, action) where action is relative to current heading BEFORE the step."""
        if not self.log_data_enabled:
            return

        dx, dy = self.direction  # current heading BEFORE applying intent

        # action encoding: 0=straight, 1=left, 2=right
        action = 0
        if intent == (-dy, dx):
            action = 1
        elif intent == (dy, -dx):
            action = 2

        state = self.get_state()
        self.writer.writerow(state + [action])
        self.csv.flush()

    def step(self, intent):
        """Apply one snake tick."""
        self.direction = intent
        new_head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])

        will_eat = (new_head == self.food)
        if self._if_collision(new_head, will_eat=will_eat):
            return True  # game over

        self.head = new_head
        self.snake.insert(0, self.head)

        if will_eat:
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        return False

    def run(self, ui=None, fps=60, tps=5):
        """High-FPS input polling with fixed-rate snake ticks."""
        dt = 0.0
        step_time = 1.0 / tps

        while True:
            # Pump input at high rate; buffer latest keypress in self.pending_dir
            self._pump_input()

            # Time accumulation
            if ui:
                dt += ui.clock.tick(fps) / 1000.0
            else:
                dt += 1.0 / fps

            # Fixed-rate update ticks
            while dt >= step_time:
                dt -= step_time

                # Consume buffered input exactly once per tick
                intent = self.direction
                if self.pending_dir and not (self.pending_dir[0] == -self.direction[0] and self.pending_dir[1] == -self.direction[1]):
                    intent = self.pending_dir
                self.pending_dir = None

                # Log state + action (relative to pre-step direction)
                self.log_data(intent)

                # Apply step
                game_over = self.step(intent)
                if game_over:
                    print(f"Collision! Final Score: {self.score}")
                    self.close()
                    return

            if ui:
                ui.render()


# --- THE OBSERVER UI ---
class SnakeUI:
    def __init__(self, engine, cell_size=30):
        pygame.init()
        self.engine = engine
        self.cell_size = cell_size

        self.width = engine.grid_w * cell_size
        self.height = engine.grid_h * cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Training: Snake Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

    def render(self):
        self.screen.fill((15, 15, 15))

        # Food
        fx, fy = self.engine.food
        pygame.draw.rect(
            self.screen,
            (255, 50, 50),
            (fx * self.cell_size, fy * self.cell_size, self.cell_size - 1, self.cell_size - 1),
        )

        # Snake
        for i, (x, y) in enumerate(self.engine.snake):
            color = (0, 255, 100) if i == 0 else (0, 180, 70)
            pygame.draw.rect(
                self.screen,
                color,
                (x * self.cell_size, y * self.cell_size, self.cell_size - 1, self.cell_size - 1),
            )

        # Score
        score_surf = self.font.render(f"Score: {self.engine.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))

        pygame.display.flip()


if __name__ == "__main__":
    game_engine = SnakeEngine(20, 15, log_data=True)
    game_ui = SnakeUI(game_engine)
    game_engine.run(ui=game_ui, fps=60, tps=5)
