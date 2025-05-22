import pygame
import random
from textblob import TextBlob
import pandas as pd

# Initialize Pygame
pygame.init()

# Load the dataset (replace with the correct path to your dataset)
df = pd.read_csv('./cleaned/twitter_cleaned_dataset.csv')

# Function to classify a tweet as a troll based on sentiment or keywords
def is_troll(tweet):
    # Check if the tweet is a valid string and not NaN
    if isinstance(tweet, str):
        analysis = TextBlob(tweet)
        sentiment = analysis.sentiment.polarity
        return sentiment < -0.2  # Threshold for negativity, adjust as needed
    else:
        return False  # If not a valid string, return False (not a troll)

# Ensure 'cleaned_text' column contains only strings, filter out non-string entries
df = df[df['cleaned_text'].apply(lambda x: isinstance(x, str))]

# Now apply the is_troll function to filter troll tweets
troll_tweets = df[df['cleaned_text'].apply(is_troll)]['cleaned_text'].tolist()


# Set up the game window
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Whack-A-Troll")

# Define some colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Define the font
font = pygame.font.SysFont('Arial', 24)

# Game variables
score = 0
trolls = []

# Define a Troll class to represent each appearing troll tweet
class Troll:
    def __init__(self, tweet):
        self.tweet = tweet
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.size = 50
        self.active = True
        self.time_to_live = random.randint(2000, 5000)  # Time in milliseconds before disappearing

    def draw(self):
        if self.active:
            pygame.draw.circle(screen, RED, (self.x, self.y), self.size)
            tweet_text = font.render(self.tweet[:30] + '...', True, WHITE)
            screen.blit(tweet_text, (self.x - 40, self.y - 10))

    def is_hit(self, pos):
        return (self.x - pos[0]) ** 2 + (self.y - pos[1]) ** 2 <= self.size ** 2

# Game loop
running = True
clock = pygame.time.Clock()

# Main game loop
while running:
    screen.fill(BLACK)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle mouse click (whack a troll)
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for troll in trolls:
                if troll.is_hit(mouse_pos):
                    troll.active = False
                    score += 1

    # Spawn new trolls
    if len(trolls) < 5:  # Limit the number of active trolls on screen
        troll_tweet = random.choice(troll_tweets)
        new_troll = Troll(troll_tweet)
        trolls.append(new_troll)

    # Draw active trolls and update their time
    for troll in trolls:
        if troll.active:
            troll.draw()

    # Remove inactive trolls
    trolls = [troll for troll in trolls if troll.active]

    # Display the score
    score_text = font.render(f'Score: {score}', True, WHITE)
    screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.update()

    # Control game speed (FPS)
    clock.tick(10)

pygame.quit()
