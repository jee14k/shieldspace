import pandas as pd

# Load the datasets
racism_df = pd.read_csv("cleaned/racism_dataset.csv")
sexism_df = pd.read_csv("cleaned/sexism_dataset.csv")

# Combine the datasets
combined_df = pd.concat([racism_df, sexism_df], ignore_index=True)

# Convert cleaned_text column to lowercase
if 'cleaned_text' in combined_df.columns:
    combined_df['cleaned_text'] = combined_df['cleaned_text'].str.lower()
    #combined_df = combined_df[combined_df['cleaned_text'].str.len() >= 3]

combined_df = combined_df[['formatted_id', 'cleaned_text', 'Annotation', 'oh_label', 'users', 'hashtags', 'links']]

# Save the merged dataset
combined_df.to_csv("cleaned/twitter_racismandsexism_dataset.csv", index=False)

print("Datasets merged, cleaned_text converted to lowercase, and saved as twitter_racismandsexism_dataset.csv")