import pygame
import pandas as pd
import random

# Load dataset
df = pd.read_csv("./cleaned/twitter_cleaned_dataset.csv")  # Ensure this file is in the same directory
df_filtered = df.dropna(subset=["cleaned_text"])  # Remove missing text entries

# Extract relevant data (tweet text and labels)
tweet_scenarios = df_filtered[["cleaned_text", "oh_label"]].values.tolist()

# Shuffle and pick 5 tweets
random.shuffle(tweet_scenarios)
tweet_scenarios = tweet_scenarios[:5]

# Format the data for the game structure
game_scenarios = []
for tweet, label in tweet_scenarios:
    game_scenarios.append({
        "text": tweet,
        "options": [
            ("Cyberbullying", label == 1.0),
            ("Not Cyberbullying", label == 0.0)
        ],
        "feedback": "Correct! Reporting and blocking helps!" if label == 1.0 else "Great! Not all negative messages are cyberbullying."
    })

# Initialize pygame
pygame.init()

# Game screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cyberbullying Escape Room")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)

# Fonts
font = pygame.font.Font(None, 30)
title_font = pygame.font.Font(None, 40)

# Game variables
current_scenario = 0
score = 0
running = True

def draw_text(text, x, y, color=BLACK, font_size=30, wrap_width=700):
    """Function to draw wrapped text on the screen."""
    font = pygame.font.Font(None, font_size)
    words = text.split(' ')
    lines = []
    line = ''
    
    for word in words:
        if font.size(line + word)[0] < wrap_width:
            line += word + ' '
        else:
            lines.append(line)
            line = word + ' '
    lines.append(line)
    
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, color)
        screen.blit(text_surface, (x, y + i * 30))

# Main game loop
while running and current_scenario < len(game_scenarios):
    screen.fill(WHITE)
    
    # Display title
    draw_text("Cyberbullying Escape Room", 250, 50, BLUE, 40)
    
    # Get current scenario
    scenario = game_scenarios[current_scenario]
    
    # Display the tweet
    draw_text(f"Tweet: {scenario['text']}", 50, 150, BLACK, 30)
    
    # Draw answer buttons
    buttons = []
    y_offset = 300
    
    for index, (option_text, is_correct) in enumerate(scenario["options"]):
        color = GREEN if is_correct else RED
        button_rect = pygame.draw.rect(screen, color, (150, y_offset + (index * 60), 500, 50))
        buttons.append((button_rect, is_correct))
        draw_text(option_text, 160, y_offset + (index * 60) + 10, WHITE, 25)
    
    pygame.display.flip()
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            for button_rect, is_correct in buttons:
                if button_rect.collidepoint(x, y):
                    if is_correct:
                        score += 1
                    
                    # Show feedback
                    screen.fill(WHITE)
                    draw_text(scenario["feedback"], 100, 250, BLACK, 30)
                    pygame.display.flip()
                    pygame.time.delay(2000)
                    
                    # Move to next scenario
                    current_scenario += 1

# Show final score
screen.fill(WHITE)
draw_text(f"Game Over! Your Score: {score}/{len(game_scenarios)}", 250, 250, BLACK, 40)
pygame.display.flip()
pygame.time.delay(3000)

pygame.quit()
