# preprocess.py

import pandas as pd
import re
from langdetect import detect, LangDetectException

def clean_text(text):
    """Clean review text by removing URLS, emails, excessive whitespace

    Input: 
    text - the review text to clean

    Outputs:
    str: the cleaned review text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lower for uniformity
    text = str(text).lower()
    
    # Remove URLs using regex
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Remove excessive whitespace by replacing with single whitespace where there is trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_language(text):
    """Detect language of text"""
    try:
        if pd.isna(text) or len(str(text).strip()) < 10:
            return 'unknown'
        return detect(str(text))
    except LangDetectException:
        return 'unknown'

def preprocess_uber_reviews(input_path, output_path):
    """
    preprocess_uber_reviews by loading, cleaning, and filtering the data.

    - No language detection due to unreliability on short informal text
    - Data is labelled as English, but contains non-english text
    - Assumes location of the datasets hardcoded, doesn't handle if it doesn't exist 
    - Assumes there is a column named "review_description"

    1. Load from csv pd.read_csv()
    2. Remove rows with missing descriptions
    3. Clean text by removing URLS, emails, and excessive whitespace
    4. Calculate word count for each review
    5. Removes duplicate reviews  
    6. Removes less than 5 word reviews
    6. Saves the cleaned dataset to uber_reviews_cleaned.csv

    Inputs:
    input_path (str): Path to uber_reviews.csv
    output_path (str): Path to the cleaned CSV uber_reviews_cleaned.csv

    Outputs:
    pd.df_clean: the dataframe of cleaned processed reviews
    """
    print("="*50)
    print("PREPROCESSING UBER REVIEWS")
    print("="*50)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Original size: {len(df):,} reviews")
    
    # 2. Remove missing reviews
    print("\n2. Removing missing reviews...")
    df = df.dropna(subset=['review_description'])
    print(f"   After removing nulls: {len(df):,} reviews")
    
    # 3. Clean text
    print("\n3. Cleaning text...")
    df['review_clean'] = df['review_description'].apply(clean_text)
    
    # 4. Calculate word count
    df['word_count'] = df['review_clean'].str.split().str.len()
    
    # 5. Remove short reviews
    review_length_limit = 5     ### limit review length ###
    print(f"\n4. Removing short reviews so reviews have better context / (usefulness) (< {review_length_limit})...") 
    # 1 word reviews provide little to draw conclusions from and bloat the 
    # dataset a lot, nearly 50% of reviews!

    # display changes
    before = len(df)
    df = df[df['word_count'] >= review_length_limit]
    removed = before - len(df)
    print(f"   Removed: {removed:,} reviews ({removed/before*100:.1f}%)")
    print(f"   Remaining: {len(df):,} reviews")
    
    # 6. Remove duplicates
    print("\n5. Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates(subset=['review_clean'])
    removed = before - len(df)
    print(f"   Removed: {removed:,} duplicates")
    print(f"   Remaining: {len(df):,} reviews")
    
    # 7. Final dataset
    df_clean = df[['review_clean', 'rating', 'word_count']].copy()
    df_clean.rename(columns={'review_clean': 'review'}, inplace=True)
    df_clean = df_clean.reset_index(drop=True)
    
    # 8. Save
    print(f"\n6. Saving to {output_path}...")
    df_clean.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"\nFinal dataset: {len(df_clean):,} reviews")
    print(f"Quality filters: word_count >= 5, duplicates removed") 
    # while this does remove a some legitimate reviews which would provide use in classification
    # it also allows us to find a higher total amount of useful reviews, after seeing the results of 1, 2, 3, 4, 5 
    # it showed the most amount of formative reviews without seeming excessive in data removal
    
    print("\nRating distribution:")
    rating_dist = df_clean['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = count / len(df_clean) * 100
        print(f"  {rating}{"✭"*rating}: {count:,} ({percentage:.1f}%)")
    
    print("\nWord count statistics:")
    print(f"  Mean: {df_clean['word_count'].mean():.1f} words")
    print(f"  Median: {df_clean['word_count'].median():.1f} words")
    print(f"  Min: {df_clean['word_count'].min()} words")
    print(f"  Max: {df_clean['word_count'].max()} words")

    print("\nVerify New Data:")
    print(f"  Short reviews: {df_clean[df_clean['word_count'] < 5]}")
    print(f"  Null values: {df_clean.isnull().sum().to_dict()}")
    print(f"  Duplicate reviews: {df_clean.duplicated(subset=['review']).sum()}")
    # lang detection takes 5+ mins so leaving it commented for now 
    #df_clean['detected_lang'] = df_clean['review'].apply(detect_language)
    #print(f"  Detected languages:\n {df_clean['detected_lang'].value_counts( )}")
    
    # Sample reviews from each rating
    print("\n" + "="*50)
    print("SAMPLE CLEANED REVIEWS")
    print("="*50)
    for rating in [1,2,3,4,5]:
        if len(df_clean[df_clean['rating'] == rating]) > 0:
            sample = df_clean[df_clean['rating'] == rating].sample(min(2, len(df_clean[df_clean['rating'] == rating])))
            print(f"\n{rating} {"✭" * rating} REVIEWS:")
            for index, row in sample.iterrows():
                print(f"  • ({row['word_count']} words) {row['review'][:100]}")
    
    # Note about language
    print("Language detection not applied due to unreliability on short")
    print("informal text. The Uber Reviews Dataset is from the Indian market, labeled as English.")
    print(" ...Manual annotation phase will identify any non-English reviews")
    
    return df_clean

if __name__ == "__main__":
    input_file = "multitag/data/uber_reviews.csv"
    output_file = "multitag/data/uber_reviews_cleaned.csv"
    
    df_clean = preprocess_uber_reviews(input_file, output_file)
    print("\nPreprocessing complete!")
    print(f"Clean dataset: {len(df_clean):,} reviews ready for sampling")