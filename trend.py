import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import pymannkendall as mk

df = pd.read_csv("put_your_directory_here")

# Get your dataframe to have columns 'year' and 'label'(sentiment)

total_sent = df.groupby('year')['label'].value_counts(normalize=True).to_frame()
temp_list = []
for sent in ['positive', 'neutral', 'negative']:
    temp = mk.original_test(total_sent[total_sent.index.get_level_values(1) == sent])
    temp_dict = temp._asdict()
    temp_dict['sentiment']  = sent
    temp_list.append(temp_dict)
pd.DataFrame(temp_list).set_index('sentiment').to_csv('./trend_by_sentiment.csv')
