import pygame
import random
import json
import os

# Load words from JSON
with open("categorized_words.json", "r") as f:
    data = json.load(f)
    good_words = [{"word": word, "type": "good"} for word in data["good_words"]]
    bad_words = [{"word": word, "type": "bad"} for word in data["bad_words"]]
    all_words = good_words + bad_words

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Words!")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Comic Sans MS", 28)
big_font = pygame.font.SysFont("Comic Sans MS", 48, bold=True)

# Load assets
basket_img = pygame.image.load("basket.png")
basket_img = pygame.transform.scale(basket_img, (100, 40))
heart_img = pygame.image.load("heart.png")
heart_img = pygame.transform.scale(heart_img, (32, 32))
start_icon = pygame.image.load("start_icon.png")
start_icon = pygame.transform.scale(start_icon, (300, 300))

def draw_gradient_background():
    for y in range(HEIGHT):
        color = (135 + y // 10, 206 - y // 15, 250)
        pygame.draw.line(screen, color, (0, y), (WIDTH, y))

def spawn_word():
    word_data = random.choice(all_words)
    color = (0, 120, 255) if word_data["type"] == "good" else (255, 70, 70)
    word_surface = font.render(word_data["word"], True, color)
    rect = word_surface.get_rect()
    rect.topleft = (random.randint(0, WIDTH - rect.width), 0)
    return {"surface": word_surface, "rect": rect, "data": word_data, "color": color}

def start_screen():
    screen.fill((255, 255, 255))
    title = big_font.render("Catch the Words!", True, (0, 100, 200))
    prompt = font.render("Press any key to start", True, (80, 80, 80))

    # Fade-in effect
    alpha_surf = pygame.Surface((WIDTH, HEIGHT))
    alpha_surf.fill((255, 255, 255))
    for alpha in range(255, -1, -5):
        screen.fill((255, 255, 255))
        screen.blit(start_icon, (WIDTH // 2 - 150, HEIGHT // 6))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 + 100))
        screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, HEIGHT // 2 + 160))
        alpha_surf.set_alpha(alpha)
        screen.blit(alpha_surf, (0, 0))
        pygame.display.flip()
        pygame.time.delay(30)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                waiting = False

def game_loop():
    basket = pygame.Rect(WIDTH // 2 - 50, HEIGHT - 60, 100, 40)
    basket_speed = 10
    falling_words = []
    score = 0
    lives = 3
    spawn_event = pygame.USEREVENT + 1
    pygame.time.set_timer(spawn_event, 1500)
    total_time = 120
    start_ticks = pygame.time.get_ticks()

    running = True
    while running:
        draw_gradient_background()
        seconds_passed = (pygame.time.get_ticks() - start_ticks) // 1000
        remaining_time = max(0, total_time - seconds_passed)
        fall_speed = 3 + seconds_passed // 30

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == spawn_event:
                falling_words.append(spawn_word())

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and basket.left > 0:
            basket.move_ip(-basket_speed, 0)
        if keys[pygame.K_RIGHT] and basket.right < WIDTH:
            basket.move_ip(basket_speed, 0)

        for word in falling_words[:]:
            word["rect"].y += fall_speed
            screen.blit(word["surface"], word["rect"])
            if word["rect"].colliderect(basket):
                if word["data"]["type"] == "good":
                    score += 1
                else:
                    lives -= 1
                falling_words.remove(word)
            elif word["rect"].top > HEIGHT:
                falling_words.remove(word)

        score = max(0, score)

        screen.blit(basket_img, basket)

        pygame.draw.circle(screen, (255, 255, 255), (100, 50), 45)
        pygame.draw.circle(screen, (0, 120, 255), (100, 50), 48, 5)
        score_text = big_font.render(str(score), True, (0, 100, 200))
        screen.blit(score_text, (100 - score_text.get_width() // 2, 50 - score_text.get_height() // 2))

        for i in range(lives):
            screen.blit(heart_img, (WIDTH - 40 * (i + 1), 20))

        time_color = (0, 200, 0) if remaining_time > 30 else (255, 100, 100)
        timer_text = font.render(f"Time: {remaining_time}s", True, time_color)
        screen.blit(timer_text, (WIDTH // 2 - timer_text.get_width() // 2, 20))

        if lives <= 0 or remaining_time <= 0:
            over_text = big_font.render("Game Over!", True, (200, 0, 0))
            screen.blit(over_text, (WIDTH//2 - over_text.get_width()//2, HEIGHT//2))
            prompt = font.render("Press R to restart or Q to quit", True, (255, 255, 255))
            screen.blit(prompt, (WIDTH//2 - prompt.get_width()//2, HEIGHT//2 + 50))
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            return True
                        elif event.key == pygame.K_q:
                            return False

        pygame.display.flip()
        clock.tick(60)

# Start and main game loop
start_screen()
while True:
    if not game_loop():
        break
pygame.quit()
