import pandas as pd
import graphlab

r_cols = ['name','Row Labels','Transactions']
ratings = pd.read_csv('customer_ratings.csv',encoding = 'latin-1')

#print(ratings.head())

ratings_base = pd.read_csv('base.csv',encoding = 'latin-1')
ratings_test = pd.read_csv('ratings_test.csv',encoding = 'latin-1')

#print(ratings_base.shape,ratings_test.shape)

train_data  = graphlab.SFrame(ratings_base)
test_data = graph.lab.SFrame(ratings_test)

#popularity_recomm = popularity_


item_sim_model = graphlab.item_similarity_recommender.create(train_data,user_id='name',item_id='Row Labels',target='Transactions',similarity_type = 'pearson')
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=1)
print(item_sim_recomm.head(20))
