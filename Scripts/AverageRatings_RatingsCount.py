import pandas as pd
import sys, csv ,operator
df = pd.read_csv('/Users/keyadesai/Desktop/reco/ratings.csv')



#print(len(df.groupby('book_id').groups.keys()))

dic = df.groupby('book_id')[['rating']].mean()
dic1 = df.groupby('book_id')[['rating']].count()


file = '/Users/keyadesai/Desktop/reco/avg_ratings1.csv'
dic.to_csv(file, sep=',')
fd1 = pd.read_csv('/Users/keyadesai/Desktop/reco/avg_ratings1.csv')

'''
file = '/Users/keyadesai/Desktop/reco/avg_ratings1.csv'
dic1.to_csv(file,sep =',')

'''