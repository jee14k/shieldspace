import pygame
import random
import pandas as pd

twitter_data = pd.read_csv("twitter_cleaned_dataset.csv")

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cyberbullying Awareness Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Fonts
font = pygame.font.Font(None, 30)
title_font = pygame.font.Font(None, 40)

# Load cleaned text and labels
df = twitter_data.dropna(subset=["cleaned_text"])  # Remove rows with missing text
tweets = df[["cleaned_text", "oh_label"]].values.tolist()

# Function to draw text
def draw_text(text, x, y, color=BLACK, font_size=30):
    font = pygame.font.Font(None, font_size)
    words = text.split(' ')
    lines = []
    line = ''
    for word in words:
        if font.size(line + word)[0] < WIDTH - 40:
            line += word + ' '
        else:
            lines.append(line)
            line = word + ' '
    lines.append(line)
    
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, color)
        screen.blit(text_surface, (x, y + i * 30))

# Game loop
running = True
score = 0
rounds = 5
current_round = 0
current_tweet, current_label = random.choice(tweets)

while running and current_round < rounds:
    screen.fill(WHITE)
    
    # Display title
    draw_text("Cyberbullying Awareness Game", 200, 50, BLUE, 40)
    
    # Display tweet
    draw_text(f"Tweet: {current_tweet}", 50, 150, BLACK, 30)
    
    # Draw buttons
    pygame.draw.rect(screen, GREEN, (150, 400, 200, 50))
    pygame.draw.rect(screen, RED, (450, 400, 200, 50))
    
    draw_text("Cyberbullying", 180, 410, WHITE, 30)
    draw_text("Not Cyberbullying", 460, 410, WHITE, 30)
    
    pygame.display.flip()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            # Check if "Cyberbullying" button was clicked
            if 150 <= x <= 350 and 400 <= y <= 450:
                user_choice = 1.0
            # Check if "Not Cyberbullying" button was clicked
            elif 450 <= x <= 650 and 400 <= y <= 450:
                user_choice = 0.0
            else:
                continue
            
            # Check answer
            if user_choice == current_label:
                score += 1
            
            # Load next tweet
            current_round += 1
            if current_round < rounds:
                current_tweet, current_label = random.choice(tweets)

# Show final score
screen.fill(WHITE)
draw_text(f"Game Over! Your Score: {score}/{rounds}", 250, 250, BLACK, 40)
pygame.display.flip()
pygame.time.delay(3000)

pygame.quit()
