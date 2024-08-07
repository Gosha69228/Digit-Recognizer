import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# train_dat = pd.read_csv('train.csv')
# test_dat = pd.read_csv('test.csv')
# df = pd.concat([train_dat, test_dat], ignore_index=True, sort=False)
# df.to_csv('output.csv', index=False)

input_file = "kag.csv"
df = pd.read_csv(input_file)

df = df.drop(columns=['Id'])

df['ImageId'] = range(1, len(df) + 1)
cols = ['ImageId'] + [col for col in df.columns if col != 'ImageId']
df = df[cols]

df.to_csv('kag2.csv', index=False)