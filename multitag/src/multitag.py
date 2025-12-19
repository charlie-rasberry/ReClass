# multitag.py
# This app enables manual annotation of reviews in the Uber dataset, for training with 
# to achieve review classifications with multi task deep learning

import tkinter as tk
from tkinter import ttk
import pandas as pd
# import langdetect
import os

class MultiTag:
    def __init__(self):
  
        self.binary_map = {
            '1': 'Yes',
            '0': 'No'
        }
        
        self.aspect_map = {
            'A': 'Driver',
            'S': 'App', 
            'D': 'Pricing',
            'F': 'Service',
            'G': 'Payment',
            'H': 'General'
        }
        
        self.sentiment_map = {
            'A': 'Positive',
            'S': 'Neutral',
            'D': 'Negative'
        }


        self.root = tk.Tk()
        # root.geometry("400x300")
        self.active_column = 0  # used for highlighting the current column 
        self.btn_width = 15 # button width
        self.number_of_aspects = 6  # number of aspect buttons
        self.root.title("MultiTag")

        #self.display_review = tk.Text(self.root, height=20, width=100, wrap='word')
        #self.display_review.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Colors for active label
        self.color_incomplete = "#003366"
        self.color_complete = "#00AA00"

        # Paths
        tagged_path = "multitag/data/uber_reviews_tagged.csv"
        sampled_path = "multitag/data/uber_reviews_sampled.csv"
        # self.load_review_data("data/uber_reviews_sampled.csv")
        # self.load_review_data("data/uber_reviews_tagged.csv")
        if not os.path.exists(tagged_path):
            print(f"Tagged file did not exist, making one at: {sampled_path}")
            sampled_df = pd.read_csv(sampled_path, low_memory=False)
            sampled_df.to_csv(tagged_path, index=False)
        self.load_review_data(tagged_path)


        # =============== GUI Elements ====================

        # highlight for the current box
        self.highlight = tk.Frame(self.root, bg="#003366", height=20, width=130)
        self.highlight.grid(row=11, column=0)

        # ROW 0: Progress indication
        self.progress_label = ttk.Label(
            self.root, 
            text="Loading...", 
            font=("Arial", 12, "bold")
        )
        self.progress_label.grid(row=0, column=0, columnspan=4, pady=(5, 0))

        # ROW 1: Review display
        self.display_review = tk.Text(self.root, height=18, width=100, wrap='word', font=("Arial", 11))
        self.display_review.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        # ROW 2: Status label
        self.status_label = ttk.Label(
            self.root, 
            text="Fill in all fields...", 
            font=("Arial", 10),
            foreground="gray"
        )

        self.status_label.grid(row=2, column=0, columnspan=4, pady=(0, 5))


        #   Labels ROW 3
        ttk.Label(self.root, text="Feature Request ? 1 (yes), 0 (no)").grid(row=3, column=0, pady=(5, 2))
        ttk.Label(self.root, text="Bug Report ? 1 (yes), 0 (no)").grid(row= 3, column=1, pady=(5, 2))
        ttk.Label(self.root, text="Aspect ? A/S/D/F/G/H/J/K/L ").grid(row= 3, column=2, pady=(5, 2))
        ttk.Label(self.root, text="Aspect Sentiment ? A/S/D").grid(row= 3, column=3, pady=(5, 2))

        # ROW 4 |Buttons| 
        # Feature Requests
        self.feature_true = ttk.Button(self.root, text="1",command=lambda: self.feature_pressed("1"), width= self.btn_width).grid(row=4, column=0, pady=2)
        self.feature_false = ttk.Button(self.root, text="0",command=lambda: self.feature_pressed("0"), width= self.btn_width).grid(row=5, column=0, pady=2)
        # Bug Reports
        self.bug_true = ttk.Button(self.root, text="1",command=lambda: self.bug_pressed("1"), width= self.btn_width).grid(row=4, column=1, pady=2)
        self.bug_false = ttk.Button(self.root, text="0",command=lambda: self.bug_pressed("0"), width= self.btn_width).grid(row=5, column=1, pady=2)
        # Aspect Buttons
        self.aspect_a = ttk.Button(self.root, text="A: Driver",command=lambda: self.aspect_pressed("A"), width= self.btn_width).grid(row=4, column=2, pady=2)
        self.aspect_s = ttk.Button(self.root, text="S: App", command=lambda: self.aspect_pressed("S"), width= self.btn_width).grid(row=5, column=2, pady=2)
        self.aspect_d = ttk.Button(self.root, text="D: Pricing", command=lambda: self.aspect_pressed("D"), width= self.btn_width).grid(row=6, column=2, pady=2)
        self.aspect_f = ttk.Button(self.root, text="F: Service", command=lambda: self.aspect_pressed("F"), width= self.btn_width).grid(row=7, column=2, pady=2)
        self.aspect_g = ttk.Button(self.root, text="G: Payment", command=lambda: self.aspect_pressed("G"), width= self.btn_width).grid(row=8, column=2, pady=2)
        self.aspect_h = ttk.Button(self.root, text="H: General", command=lambda: self.aspect_pressed("H"), width= self.btn_width).grid(row=9, column=2, pady=2)
        # self.aspect_j = ttk.Button(self.root, text="J: ASPECT HERE", command=lambda: self.aspect_pressed("J"), width= self.btn_width).grid(row=4, column=2, pady=2)
        # self.aspect_k = ttk.Button(self.root, text="K: ASPECT HERE", command=lambda: self.aspect_pressed("K"), width= self.btn_width).grid(row=4, column=2, pady=2)
        # self.aspect_l = ttk.Button(self.root, text="L: ASPECT HERE", command=lambda: self.aspect_pressed("L"), width= self.btn_width).grid(row=4, column=2, pady=2)
        # Aspect sentiment buttons
        self.aspect_positive = ttk.Button(self.root, text="A: Positive", command=lambda: self.sentiment_pressed("A"), width= self.btn_width).grid(row=4, column=3, pady=2)
        self.aspect_neutral = ttk.Button(self.root, text="S: Neutral", command=lambda: self.sentiment_pressed("S"), width= self.btn_width).grid(row=5, column=3, pady=2)
        self.aspect_negative = ttk.Button(self.root, text="D: Negative", command=lambda: self.sentiment_pressed("D"), width= self.btn_width).grid(row=6, column=3, pady=2)

        # Highlight box - positioned below buttons
        # self.highlight = tk.Frame(self.root, bg=self.color_incomplete, height=20, width=130)
        self.highlight.grid(row=10, column=0, pady=(5, 5))

        #   Key bindings
        self.root.bind("q", self.quit_app)
        self.root.bind("<Return>", self.try_submit)
        self.root.bind("1", self.handle_key)
        self.root.bind("0", self.handle_key)
        self.root.bind("a", self.handle_key)
        self.root.bind("s", self.handle_key)
        self.root.bind("d", self.handle_key)
        self.root.bind("f", self.handle_key)
        self.root.bind("g", self.handle_key)
        self.root.bind("h", self.handle_key)
        # self.root.bind("j", self.handle_key)
        # self.root.bind("k", self.handle_key)
        # self.root.bind("l", self.handle_key)


    
        self.display_next_review()
        #   self.save_tags("data/uber_reviews_tagged.csv")
        self.root.mainloop()

    def handle_key(self, event):
        key = event.char
    
        # Column 0 or 1: feature/bug (1 and 0)
        if key in ['1', '0']:
            if self.active_column == 0:
                self.feature_pressed(key)
            elif self.active_column == 1:
                self.bug_pressed(key)
        # Column 2: aspects (a,s,d,f,g,h,j,k,l)
        elif key in 'asdfgh' and self.active_column == 2:
            self.aspect_pressed(key.upper())
        # Column 3: sentiment (a,s,d)
        elif key in 'asd' and self.active_column == 3:
            self.sentiment_pressed(key.upper())

    def update_status(self):
        """Update status label and highlight color based on completion state"""
        if self.all_labels_complete():
            self.highlight.configure(bg=self.color_complete)
            self.status_label.configure(
                text="Complete Tag [ENTER] | Quit [q]", 
                foreground="green",
                font=("Arial", 10, "bold")
            )
        else:
            self.highlight.configure(bg=self.color_incomplete)
            self.status_label.configure(
                text="Fill in all fields...", 
                foreground="gray",
                font=("Arial", 10)
            )

    def update_progress(self):
        """Update progress counter"""
        tagged_count = (self.review_data['tagged'] == 1).sum()
        total_count = len(self.review_data)
        remaining = total_count - tagged_count
        
        progress_text = f"Progress: {tagged_count} / {total_count} tagged ({remaining} remaining)"
        self.progress_label.configure(text=progress_text)
    
    def move_highlight(self, col):
        """Move the highlight box directly under the button pressed."""
        self.highlight.grid(row=10, column=col, pady=(5,5))
        self.update_status()

    # Setters
    def feature_pressed(self, value):
        self.review_data.at[self.current_review_index, "feature_request"] = self.binary_map[value]
        self.active_column = 1
        self.move_highlight(1)

    def bug_pressed(self, value):
        self.review_data.at[self.current_review_index, "bug_report"] = self.binary_map[value]
        self.active_column = 2
        self.move_highlight(2)

    def aspect_pressed(self, value):
        self.review_data.at[self.current_review_index, "aspect"] = self.aspect_map[value]
        self.active_column = 3
        self.move_highlight(3)

    def sentiment_pressed(self, value):
        self.review_data.at[self.current_review_index, "aspect_sentiment"] = self.sentiment_map[value]
        self.active_column = 0  # Reset for next review
        self.update_status()


    def load_review_data(self, data_path):
        """Load review data from a CSV file."""
        self.review_data = pd.read_csv(data_path, low_memory=False)
        if "tagged" not in self.review_data.columns:
            self.review_data["tagged"] = 0              # Initialize tagged column if not present
        if "feature_request" not in self.review_data.columns:
            self.review_data["feature_request"] = ""    # Initialize feature_request column if not present
        if "bug_report" not in self.review_data.columns:
            self.review_data["bug_report"] = ""         # Initialize bug_report column if not present
        if "aspect" not in self.review_data.columns:
            self.review_data["aspect"] = ""             # Initialize aspect column if not present
        if "aspect_sentiment" not in self.review_data.columns:
            self.review_data["aspect_sentiment"] = ""   # Initialize aspect_sentiment column if not present
        print(f"Loaded {len(self.review_data)} reviews from {data_path}")
    
    def display_next_review(self):
        """Display the next review in the text box."""
        self.current_review_index = self.get_current_review_index()
        if self.current_review_index < len(self.review_data):
            review = self.review_data.iloc[self.current_review_index]

            self.review_data.at[self.current_review_index, "feature_request"] = ""
            self.review_data.at[self.current_review_index, "bug_report"] = ""
            self.review_data.at[self.current_review_index, "aspect"] = ""
            self.review_data.at[self.current_review_index, "aspect_sentiment"] = ""

            self.display_review.delete(1.0, tk.END)  # Clear the text box
            self.display_review.insert(tk.END, review["review"])  # Display the review text
            # self.current_review_index += 1
            # Mark as tagged
            #   self.review_data.at[self.current_review_index - 1, "tagged"] = 1
            self.active_column = 0  # reset to start at feature request
            self.highlight.grid(row=10, column=0, pady=(5, 5))
            self.highlight.configure(bg=self.color_incomplete)
            self.status_label.configure(
                text="Fill in all fields...", 
                foreground="gray",
                font=("Arial", 10)
            )
            self.update_progress()
            self.update_progress()
            
        else:
            print("No more reviews to display. DONE 	☉ ‿ ⚆")

    def submit_tag(self):
        self.review_data.at[self.current_review_index, "tagged"] = 1
        self.save_tags("multitag/data/uber_reviews_tagged.csv")
        self.display_next_review()

    def try_submit(self, event):
        """Try to submit current review if all labels complete."""
        if self.all_labels_complete():
            self.submit_tag()
            print(f"Review {self.current_review_index + 1} tagged")
        else:
            print("      ☠      Complete all fields first!     ☠      ")
            self.status_label.configure(
                text="      ☠      Complete all fields first!     ☠      ", 
                foreground="red",
                font=("Arial", 10, "bold")
            )
            self.root.after(2000, self.update_status)
    
    def all_labels_complete(self):
        row = self.review_data.iloc[self.current_review_index]
        return (row["feature_request"] != "" and 
            row["bug_report"] != "" and 
            row["aspect"] != "" and 
            row["aspect_sentiment"] != "")
    
    def save_tags(self, save_path):
        """Save the tagged data to a CSV file."""
        self.review_data.to_csv(save_path, index=False)
        # print(f"Tagged data saved to {save_path}")

    def quit_app(self, event):
        tagged_count = (self.review_data['tagged'] == 1).sum()
        print(f"\n{'='*50}")
        print(f"SESSION COMPLETE")
        print(f"{'='*50}")
        print(f"Total tagged: {tagged_count} / {len(self.review_data)}")
        print(f"Saved to: multitag/data/uber_reviews_tagged.csv")
        print(f"Bye    (ʘ‿ʘ)╯")
        self.save_tags("multitag/data/uber_reviews_tagged.csv")
        self.root.destroy()

    def get_current_review_index(self):
        for i in range(len(self.review_data)):
            if self.review_data.iloc[i]["tagged"] == 0:
                return i
        return len(self.review_data)  # all reviews tagged
    
    
   
app = MultiTag()
