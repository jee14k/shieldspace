import time
import random
import pandas as pd

# Load the cleaned tweet dataset
def load_tweets(file_path):
    df = pd.read_csv(file_path)
    return df['cleaned_text'].dropna().tolist()

# Intro
def intro():
    print("\nWelcome to Empathy RPG - A Cyberbullying Awareness Game!\n")
    time.sleep(1)
    print("You will step into the shoes of different people involved in online interactions.")
    print("Your choices will shape the story and reveal the impact of cyberbullying.\n")
    time.sleep(2)

# Victim Scenario
def victim_story(tweets):
    print("\n--- SCENARIO: Victim ---\n")
    tweet = random.choice(tweets)
    print(f"You receive a mean tweet: {tweet}")
    
    choice = input("What do you do? (A: Ignore / B: Respond Kindly / C: Report) ").strip().upper()
    
    if choice == "A":
        print("You ignore the tweet, but it still affects you emotionally.")
    elif choice == "B":
        print("You respond with kindness, hoping to de-escalate the situation.")
    elif choice == "C":
        print("You report the tweet, and it gets removed. Good decision!")
    else:
        print("Invalid choice. Moving to the next scenario.")
    time.sleep(2)

# Bystander Scenario
def bystander_story(tweets):
    print("\n--- SCENARIO: Bystander ---\n")
    tweet = random.choice(tweets)
    print(f"You see a classmate getting bullied online: {tweet}")
    
    choice = input("What do you do? (A: Ignore / B: Defend the Victim / C: Report to an Adult) ").strip().upper()
    
    if choice == "A":
        print("You ignore it, but later feel guilty for not stepping in.")
    elif choice == "B":
        print("You stand up for your classmate and tell the bully to stop.")
    elif choice == "C":
        print("You report the incident to a trusted adult, helping to stop the bullying.")
    else:
        print("Invalid choice. Moving to the next scenario.")
    time.sleep(2)

# Aggressor Scenario
def aggressor_story(tweets):
    print("\n--- SCENARIO: Aggressor ---\n")
    tweet = random.choice(tweets)
    print(f"You post a sarcastic tweet: {tweet}")
    
    choice = input("What happens next? (A: Apologize / B: Justify as a Joke / C: Ignore the Reaction) ").strip().upper()
    
    if choice == "A":
        print("You apologize and learn that words can hurt, even unintentionally.")
    elif choice == "B":
        print("You justify it as a joke, but the person you targeted is still hurt.")
    elif choice == "C":
        print("You ignore it, but the person stops talking to you.")
    else:
        print("Invalid choice. Moving to the conclusion.")
    time.sleep(2)

# Conclusion
def conclusion():
    print("\n--- Game Over ---\n")
    print("Cyberbullying has real consequences. Think before you type and be kind online!")
    print("Thanks for playing Empathy RPG!")

# Main Game Loop
def play_game(file_path):
    tweets = load_tweets(file_path)
    
    intro()
    victim_story(tweets)
    bystander_story(tweets)
    aggressor_story(tweets)
    conclusion()

if __name__ == "__main__":
    # Provide the path to your cleaned_twitter_dataset.csv
    play_game("twitter_cleaned_dataset.csv")