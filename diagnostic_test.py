import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import random
import sys
import os

# --- 1. FORCE VISIBILITY ---
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

# --- 2. CONFIGURATION ---
FRAME_WIDTH, FRAME_HEIGHT = 1200, 800
LATENT_DIM = 64
DEVICE = torch.device('cpu')

print(f"SYSTEM STATUS: INITIALIZING ON {DEVICE}")


# --- 3. THE ENGINE ---
class ChineseCharacterAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


# --- 4. DATA GENERATOR ---
def generate_synthetic_glyph():
    grid = torch.zeros((1, 28 * 28))
    for _ in range(random.randint(4, 7)):
        start_idx = random.randint(0, 783)
        direction = random.choice([1, -1, 28, -28, 29, -29])
        length = random.randint(5, 15)
        for i in range(length):
            pos = start_idx + (i * direction)
            if 0 <= pos < 784:
                grid[0, pos] = random.uniform(0.7, 1.0)
    return grid.to(DEVICE)


# --- 5. RENDER PIPELINE (INTERACTIVE) ---
def run_system():
    pygame.init()
    screen = pygame.display.set_mode((FRAME_WIDTH, FRAME_HEIGHT))
    pygame.display.set_caption("KINETIC AUTOENCODER: LIVE CONTROL")
    clock = pygame.time.Clock()

    font_mono = pygame.font.SysFont("Courier New", 14, bold=True)
    font_header = pygame.font.SysFont("Arial", 24, bold=True)
    font_status = pygame.font.SysFont("Impact", 20)

    model = ChineseCharacterAE().to(DEVICE)

    # Start with a moderate learning rate
    current_lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    criterion = nn.MSELoss()

    running = True
    frame_count = 0
    input_data = generate_synthetic_glyph()

    # Control Flags
    inject_noise = False

    print("SYSTEM STATUS: LIVE CONTROL ENGAGED")

    while running:
        # A. INPUTS (CONTROL BOARD)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # 1. Reset Target
                if event.key == pygame.K_r:
                    input_data = generate_synthetic_glyph()

                # 2. Adjust Velocity (Learning Rate)
                if event.key == pygame.K_UP:
                    current_lr *= 1.5
                    # Cap max speed to prevent total crash
                    current_lr = min(current_lr, 10.0)
                    # Update Optimizer
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                if event.key == pygame.K_DOWN:
                    current_lr *= 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                # 3. Trauma Injection (Spacebar)
                if event.key == pygame.K_SPACE:
                    inject_noise = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    inject_noise = False

        # B. LOGIC
        optimizer.zero_grad()
        reconstruction, latent = model(input_data)

        # Apply Trauma if Spacebar is held
        target = input_data.clone()
        if inject_noise:
            noise = torch.randn_like(target) * 0.5
            target = target + noise

        loss = criterion(reconstruction, target)
        loss.backward()
        optimizer.step()

        delta_val = loss.item()
        latent_vec = latent[0].detach().numpy()

        # C. DRAWING
        screen.fill((5, 5, 10))

        # 1. Grids
        scale = 12
        offset_y = 200

        # Input
        input_img = target.view(28, 28).detach().numpy()  # Show what the model SEES (including noise)
        for r in range(28):
            for c in range(28):
                val = input_img[r, c]
                if val > 0.1:
                    # Trauma turns input Red, Normal is Gray
                    col = (255, 50, 50) if inject_noise else (int(val * 255), int(val * 255), int(val * 255))
                    # Clamp color values to 255
                    col = tuple(min(255, max(0, c)) for c in col)
                    pygame.draw.rect(screen, col, (50 + c * scale, offset_y + r * scale, scale - 1, scale - 1))

        # Output
        recon_img = reconstruction.view(28, 28).detach().numpy()
        for r in range(28):
            for c in range(28):
                val = recon_img[r, c]
                if val > 0.1:
                    g_val = int(val * 255)
                    pygame.draw.rect(screen, (0, g_val, 0),
                                     (450 + c * scale, offset_y + r * scale, scale - 1, scale - 1))

        # 2. METRICS
        bar_len = min(int(delta_val * 5000), 1100)
        pygame.draw.rect(screen, (50, 0, 0), (50, 650, 1100, 40))
        pygame.draw.rect(screen, (255, 0, 0), (50, 650, bar_len, 40))

        # 3. CONTROL DASHBOARD (Top Yellow Line)
        # Dynamic color for LR: Yellow = Safe, Red = Dangerous
        lr_color = (255, 255, 0) if current_lr < 1.0 else (255, 0, 0)

        status_text = f"LEARNING VELOCITY (LR): {current_lr:.6f} | NOISE: {'ACTIVE' if inject_noise else 'OFF'}"
        screen.blit(font_status.render(status_text, True, lr_color), (50, 50))

        screen.blit(font_header.render(f"LOSS: {delta_val:.6f}", True, (255, 100, 100)), (50, 610))

        # 4. LATENT
        pygame.draw.line(screen, (0, 255, 0), (850, 50), (850, 600), 2)
        for i in range(20):
            val = latent_vec[i]
            col = (0, 255, 0) if val > 0 else (0, 100, 0)
            screen.blit(font_mono.render(f"DIM_{i:02}: {val:+.4f}", True, col), (870, 100 + i * 20))

        # Instructions
        controls = "CONTROLS: [UP/DOWN] SPEED | [SPACE] TRAUMA | [R] NEW GLYPH"
        screen.blit(font_mono.render(controls, True, (100, 100, 255)), (50, 750))

        pygame.display.flip()
        frame_count += 1
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_system()