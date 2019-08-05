import pandas as pd
import numpy as np

df = pd.read_csv('AllTweets.csv')

states = pd.read_csv('states_from_internet.csv')

df_length = len(df)

df['State'] = ''

for i in range(0, df_length):
    temp_loc = df.at[i, 'Location']
    if temp_loc is not np.nan:
        for s in range(0,51):
            if any([states.at[s,'Abbreviation'] in temp_loc, states.at[s,'State'] in temp_loc]):
                df.at[i,'State'] = states.at[s,'Abbreviation']
                # this "break" here is pretty flimsy ... the point is to make sure that DC
                # is labeled as DC and not Washington.. it only works because DC comes first in
                # the alphabet..
                break

# A dataframe whose indexes are state abbreviations and values are the frequency of that state
# in our dataset
state_counts = df.State.str.split(expand=True).stack().value_counts()
# Make a count column and set all values to 0
states['count'] = 0
# Loop through every state
for s in range(0,51):
    if states.at[s,'Abbreviation'] in state_counts.index:
        states.at[s,'count'] = state_counts[states.at[s,'Abbreviation']]

# save the data
df.to_csv('AllTweets_WithStates.csv')
states.to_csv('StatesWithCounts.csv')