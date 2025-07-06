import pandas as pd



# Step 2: Concatenate them
combined_df = pd.read_csv('FC_Ageing.csv')



# Step 4: Describe
describe_df = combined_df.describe(include='all')

# Step 5: Save describe to CSV
describe_df.to_csv('describe1.csv', encoding='utf-8')
