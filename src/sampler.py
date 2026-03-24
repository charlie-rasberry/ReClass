import pandas as pd
import numpy as np

print(pd.__version__)
print(np.__version__)

path = "data/raw/uber_reviews_cleaned.csv"
sampled_path = "data/raw/uber_reviews_sampled.csv"
original_path = "data/raw/uber_reviews.csv" ### only for distribution comparison
class Sampler:
    def __init__(self, data_path, target_samples):

        self.data_path = data_path
        self.stratify_column = "rating"  # column to stratify by (another sampleset will use keyword boosting to aid feature request / bug report numbers)

        self.original_data = pd.read_csv(original_path, low_memory=False)
        self.data = pd.read_csv(self.data_path, low_memory=False)
        self.total = len(self.data)  # total number of records in the dataset

        print("="*50)
        print("SAMPLER INITIALIZED")
        print("="*50,"\n")


        print(f"Total records in dataset: {self.total}")
        print(f"Data loaded from {self.data_path}, total records: {len(self.data)}")
        #print(self.data.head())
        #print(f"\nCurrent distribution:")
        #print(self.data[self.stratify_column].value_counts().sort_index())
        #print(f"\nColumns: {self.data.columns.tolist()}")
        print(f"Percentage distribution (working data):")
        print((self.data[self.stratify_column].value_counts(normalize=True).sort_index() * 100).round(1),"\n")
        _origdist = self.original_data[self.stratify_column].value_counts(normalize=True).sort_index()
        print(f"Original Distribution from {original_path}:")
        print((_origdist*100).round(1),"\n")

        self.data.info(verbose=True)

    #   add sampling method here
    #   random sample 5000 entries with stratifiying by rating
    """
    rating
    5    57.1% (611133)
    1    26.5% (283895)
    4     7.8% (82953)
    3     4.7% (49928)
    2     3.9% (41707)
    Name: proportion, dtype: object
    """
    
    """
    IGNORE --- Left in just in case

    Sample randomly
    Redundant calculation
    Doesn't factor that the distribution changed greatly after preprocessing
    """
    def get_stratified_sample(self) -> pd.DataFrame:
           stratified_sample = (
            self.data
            .reset_index(drop=True) # remove messy indexes
            .apply(self.sample_col) # applies to each column
            .sample(n=self.target_samples, random_state=42) # 42 on sampler 4321 on any other file
            )
           return stratified_sample
        
    
    def sample_col(self, column) -> pd.DataFrame:    
        """
        IGNORE --- Left in just in case

        Randomly sample, including conflicting math, I guess I was going to stratify
        """
        samples_per_column = int(len(column) / self.total * self.target_samples) # pointless 1 *5000
        samples_per_column = max(samples_per_column,1) # also pointless
        return column.sample(n=samples_per_column, random_state=42)


    """
    original_distribution_sample()
    The main sampling method for our labelling as it 
    keeps composition of the original uber dataset, verified in 
    which is a fairer comparison, may also work better in general

    verified post preprocessing in rating_distribution.ipynb and verify_tagged_distributions.ipynb
    and raw data distribution verified at the bottom of verify_tagged_distributions.ipynb

    
    manually coded distributions taken from notebooks

    for ratings and actual number of samples 
    rating data is the whole data for a rating as we iterate
    has error handling if totals doesn't match the required amount of samples per the orig distrib
    randomise the indexes (samples) and appends to the new dataset



    """
    def original_distribution_sample(self):
        original_dist = {
            5: int(0.571 * self.target_samples), 
            1: int(0.265 * self.target_samples),  
            4: int(0.078 * self.target_samples),  
            3: int(0.047 * self.target_samples),  
            2: int(0.039 * self.target_samples)   
        }        
        print("Target Distribution =", original_dist)
        samples = []
        for rating, num_samples in original_dist.items():
            rating_data = self.data[self.data[self.stratify_column] == rating] # stratify_column = "rating"
            if len(rating_data) < num_samples:                                 # data is a pd.dataframe of the set
                print("Missing samples available for rating")
                num_samples = len(rating_data)
            sample = rating_data.sample(n = num_samples,random_state=42)
            samples.append(sample)
        original_sample = pd.concat(samples, ignore_index=True)
        return original_sample
    
    """
    sample_with_keywords()

    In order to train on more bugs and features data in 
    future this method was created
    - 2000 balanced by rating (400 per)
    - 1500 likely bugs using bug_keywords list
    - 1500 likely features using feature_keywords list

    inputs:
    outputs:
    
    """

    def sample_with_keywords(self):
        #TODO add keywords for feature classification
        print(f"\n{'='*50}")
        print("Keyword influenced / rating stratified set")
        print(f"\n{'='*50}")

        bug_keywords = ["crash","freeze", "error",
                        "stop", "doesnt work", "doesn't work","loading",
                        "blank", "stuck", "load", "broken", "break",
                        "glitch", "issue", "fix", "needs","please repair",
                        "failed", "responding"
                        ]
        feature_keywords = ["need","should","add","wish","would","benefit",
                            "please add","should have", "want", "missing",
                            "require", "suggestion", "request", "could you",
                            "include", "hope", "why not", "greatly", "option",
                            "new","system"
                            ]
        self.data['likely_bug'] = self.data['review'].apply(
            lambda x:any(keyword in str(x).lower() for keyword in bug_keywords)
        )
        self.data['likely_feature'] = self.data['review'].apply(
            lambda x: any (keyword in str(x).lower() for keyword in feature_keywords)
        )
        print(f"Reviews with bug_keywords = {self.data['likely_bug'].sum():,}")
        print(f"Reviews with feature_keywords = {self.data['likely_feature'].sum():,}")

        print(f"Sampling 2000 reviews balanced (400 per rating)...")
        base_sample = self.data.groupby(self.stratify_column).apply(
            lambda x: x.sample(n=min(400, len(x)), random_state=42),
            include_groups = False
        ).reset_index(drop=True)

        print(f"Sampling 1500 possible bug reports...")
        bugs = self.data[self.data['likely_bug'] & ~self.data.index.isin(base_sample.index)]
        bug_sample = bugs.sample(n=min(1500, len(bugs)), random_state=42)
        
        print(f"Sampling 1500 possible feature requests...")
        features = self.data[
            self.data['likely_feature'] & 
            ~self.data.index.isin(base_sample.index) &
            ~self.data.index.isin(bug_sample.index)
        ]
        feature_sample = features.sample(n=min(1500, len(features)), random_state=42)

        # Combine all samples
        keyword_sample = pd.concat([base_sample, bug_sample, feature_sample], ignore_index=True)
        
        # Drop helper columns
        keyword_sample = keyword_sample.drop(columns=['likely_bug', 'likely_feature'])

        
        
        print(f"\n Total samples: {len(keyword_sample):,}")
        return keyword_sample

    def sample_tiny_size(self):
        mini_sample = self.data.sample(200)     #   reading some samples manually
        return mini_sample

         
    
    def save_sample(self, sample_df,output_path):
        """Save sample and display statistics"""
        sample_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*50}")
        print("SAMPLE SAVED")
        print(f"{'='*50}")
        print(f"Location: {output_path}")
        print(f"Total samples: {len(sample_df):,}")
        print(f"\nDistribution:")
        for rating in sorted(sample_df[self.stratify_column].unique()):
            count = (sample_df[self.stratify_column] == rating).sum()
            pct = count / len(sample_df) * 100
            print(f"  {rating}★: {count:,} ({pct:.1f}%)")

def main():
    
    sampler = Sampler("data/raw/uber_reviews_cleaned.csv", target_samples=5000)

    # Choose sampling strategy
    print(f"\n{'='*50}")
    print("SAMPLING STRATEGY OPTIONS")
    print(f"{'='*50}")
    print("1. get_stratified_sample() stratified by current distribution")
    print("2. original_distribution_sample() stratified by the original data distribution")
    print("3. get_keyword_boosted_sample() stratified using original distribution but also using a keyword dictionary")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        sample = sampler.get_stratified_sample()
        sampler.save_sample(sample, "data/raw/uber_reviews_sampled.csv")
        
    elif choice == '2':
        sample = sampler.original_distribution_sample()
        sampler.save_sample(sample, "data/raw/uber_reviews_sampled.csv")
        
    elif choice == '3':
        sample = sampler.sample_with_keywords()
        sampler.save_sample(sample, "data/raw/uber_reviews_sampled.csv")

    elif choice == '4':
        sample = sampler.sample_tiny_size()
        sampler.save_sample(sample,"data/raw/uber_review_temp.csv")
        


if __name__ == "__main__":
    main()