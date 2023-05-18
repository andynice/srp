import pandas as pd

file_exists = False

# List of input CSV files
file1 = r"C:\Users\Toby\Desktop\Master-Data\Semester-2\SRP\data\data-Jan1.csv"
file2 = r"C:\Users\Toby\Desktop\Master-Data\Semester-2\SRP\data\data-Jan2.csv"
file3 = r"C:\Users\Toby\Desktop\Master-Data\Semester-2\SRP\data\data-Jan3.csv"
input_files = [file1, file2, file3]

# Output CSV file
output_file = 'output_Jan.csv'


# Processed text function
def process_text(text):
    # Add your text processing logic here
    processed_text = text.upper()  # Example: Convert text to uppercase

    

    return processed_text


# Process and write data for each input CSV file
for file in input_files:
    # Read input file
    data = pd.read_csv(file)
    
    data = data[data["lang"].isin(["en", "de"])]
    # filtered_df = df[~df["lang"].isin(["en", "de"])] >> the ~ specifies "not" in the expressions
    
    data = data.drop(["geo", "id", "source"], axis=1)


    # Convert "created_at" column to datetime
    data['created_at'] = pd.to_datetime(data['created_at'])

    # Remove the minutes, seconds, and timezone offset
    # data['created_at'] = data['created_at'].dt.strftime('%Y-%m-%d %H:00:00')
    data['created_at'] = data['created_at'].dt.strftime('%Y-%m-%d %H')

    # Process text column
    data['freq_count'] = data['tweet'].apply(process_text)
    
    # Write processed data to output file
    data.to_csv(output_file, mode='a', index=False, header=not file_exists)

    # Set file_exists to True after writing the header once
    file_exists = True

