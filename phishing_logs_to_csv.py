# Import the csv module to get access to quoting constants
import csv
from datasets import load_dataset
import pandas as pd

print("Loading dataset from Hugging Face...")
# Use your token if you still need it for authentication
# dataset = load_dataset("renemel/compiled-phishing-dataset", token="hf_...")
dataset = load_dataset("renemel/compiled-phishing-dataset")

print("Converting to Pandas DataFrame...")
df = dataset['train'].to_pandas()

print("Saving DataFrame to CSV with proper quoting...")
# Use the 'quoting' parameter to handle special characters correctly
# csv.QUOTE_NONNUMERIC tells it to quote all fields that aren't numbers.
# This is a very safe option.
df.to_csv("phishing_dataset_fixed.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

print("Dataset saved successfully to phishing_dataset_fixed.csv")