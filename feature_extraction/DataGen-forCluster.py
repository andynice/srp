import pandas as pd
import glob
from nltk.tokenize import word_tokenize
from collections import Counter

# Create an empty dictionary to store word frequencies
word_frequencies = {}
total_cases = {}

# Specify the directory containing your CSV files
# csv_directory = './cleaned_data'
# on the cluster, use this path
csv_directory = '/home/correa/text_cleaning/output'

# Use glob to get a list of CSV files in the directory
csv_files = glob.glob(f'{csv_directory}/*.csv')


# ###
# ### The Func
# ###

# Define a function to count word frequencies
def count_word_frequencies(text):
    words = word_tokenize(text)
    word_count = Counter(words)
    return word_count


# ###
# ### The Loop
# ###

# Loop through each CSV file, read it, and count word frequencies
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Assuming the column names are "created_at" and "clean_tweets"
    if 'clean_tweets' in df.columns:
        df['word_frequencies'] = df['clean_tweets'].apply(count_word_frequencies)

        # Update the word frequencies dictionary
        for _, row in df.iterrows():
            created_at = row['created_at']
            word_count = row['word_frequencies']

            if created_at not in word_frequencies:
                word_frequencies[created_at] = word_count
            else:
                word_frequencies[created_at].update(word_count)

# Create a DataFrame from the word frequencies dictionary
word_frequency_df = pd.DataFrame.from_dict(word_frequencies, orient='index')

# Reset the index and fill NaN values with 0
word_frequency_df = word_frequency_df.reset_index().fillna(0)

# Rename the columns
word_frequency_df = word_frequency_df.rename(columns={'index': 'created_at'})

# Save the DataFrame to a CSV file
word_frequency_df.to_csv('word_frequencies.csv', index=False)