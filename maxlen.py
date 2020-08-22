import pandas as pd
max=0
df = pd.read_csv('training_data.csv')
for i in df['Text']:
    if len(i)>max:
        max=len(i)

print(max)