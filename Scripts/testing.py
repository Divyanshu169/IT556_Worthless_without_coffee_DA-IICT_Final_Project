import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
from sklearn.model_selection import train_test_split
import csv
import warnings; warnings.simplefilter('ignore')
import math


# coding: utf-8

# In[1]:

def hybrid(userId,train_rd):
    #get_ipython().magic('matplotlib inline')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.corpus import wordnet
    from surprise import Reader, Dataset, SVD, evaluate

    import warnings; warnings.simplefilter('ignore')


    # In[2]:


    #Popularity#

    md = pd.read_csv('CustomData/FinalData.csv')

    fd = pd.read_csv('avg_ratings1.csv')



    fd[fd['rating'].notnull()]['rating'] = fd[fd['rating'].notnull()]['rating'].astype('float')
    vote_averages= fd[fd['rating'].notnull()]['rating']
    C = vote_averages.mean()


    fd1 = pd.read_csv('ratings_count.csv')


    fd1[fd1['rating'].notnull()]['rating'] = fd1[fd1['rating'].notnull()]['rating'].astype('float')
    vote_counts = fd1[fd1['rating'].notnull()]['rating']


    # In[3]:


    m = vote_counts.quantile(0.75)



    # In[4]:


    md['ratings_count'] = fd1['rating']
    md['average_rating'] = fd['rating']


    # In[28]:


    #print(md.shape)
    qualified = md[(md['ratings_count'].notnull())][['book_id','title', 'authors', 'ratings_count', 'average_rating']]

    qualified['ratings_count'] = qualified['ratings_count'].astype('float')

    qualified['average_rating'] = qualified['average_rating'].astype('float')

    #qualified.shape


    # In[29]:


    def weighted_rating(x):
        v = x['ratings_count']
        R = x['average_rating']
        return (v/(v+m) * R) + (m/(m+v) * C)


    # In[30]:


    qualified['popularity_rating'] = qualified.apply(weighted_rating, axis=1)
    #qualified['wr']
    #qualified = qualified.sort_values('popularity_rating', ascending=False).head(250)
    pop = qualified[['book_id','popularity_rating']]
    #print(qualified.shape)
    #print(pop.shape)


    # In[11]:


    ### Collaborative ##

    reader = Reader()
    ratings=train_rd
    #ratings = pd.read_csv('ratings.csv')
    #ratings.head()

    temp_ratings = ratings[0:1000]

    #print(temp_ratings)
    data = Dataset.load_from_df(temp_ratings[['user_id', 'book_id', 'rating']], reader)
    data.split(n_folds=2)


    # In[12]:


    svd = SVD()
    evaluate(svd, data, measures=['RMSE', 'MAE'])


    # In[13]:


    trainset = data.build_full_trainset()
    #svd.train(trainset)
    algo = SVD()
    algo.fit(trainset)

    ## usefule = temp_rating[rating]


    # In[14]:


#print(len(temp_ratings[temp_ratings['user_id']==userId]))


    # In[ ]:


    def get_top_n(predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            #user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n


    # In[15]:


    from collections import defaultdict
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    '''
    top_n = get_top_n(predictions, n=10000)

    #print(top_n)
    #result = pd.DataFrame(top_n)
    #print(result)
    for uid, user_ratings in top_n.items():
    
        #print(uid, [iid for (iid  , _) in user_ratings])
        for uid, iid, true_r, est, _ in predictions:
        
            temp_ratings.loc[uid]= [uid,iid,est]
        #temp_ratings[i]['cf'] = temp_ratings[(temp_ratings['user_id'] == uid)][['book_id']]
        
    '''
    count = 0
    for uid, iid, true_r, est, _ in predictions:
        
         if uid == userId:
            count = count+1
            temp_ratings.loc[len(temp_ratings)+1]= [uid,iid,est]
            #print('here')

            #print(uid)
            #temp_ratings.append([uid,iid,est],ignore_index=True)

    #print(count)
    #print(temp_ratings)



    # In[16]:


    #print(len(temp_ratings[temp_ratings['user_id']==2]))


    # In[ ]:





    # In[46]:


    ##### CONTENT ######

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.corpus import wordnet
    from surprise import Reader, Dataset, SVD, evaluate
    import csv
    import warnings; warnings.simplefilter('ignore')


    # In[48]:



    md=pd.read_csv('CustomData/FinalData.csv')
    rd=train_rd
    #rd=pd.read_csv('ratings.csv')
    md['book_id'] = md['book_id'].astype('int')
    rd['book_id'] = rd['book_id'].astype('int')
    rd['user_id'] = rd['user_id'].astype('int')
    rd['rating'] = rd['rating'].astype('int')

    #print(md.head())


    md['authors'] = md['authors'].str.replace(' ','')
    md['authors'] = md['authors'].str.lower()
    md['authors'] = md['authors'].str.replace(',',' ')

    #print(md.head())

    md['authors'] = md['authors'].apply(lambda x: [x,x])
    #print(md['authors'])

    md['Genres']=md['Genres'].str.split(';')
    #print(md['Genres'])

    md['soup'] = md['authors'] + md['Genres']
    #print(md['soup'])

    md['soup'] = md['soup'].str.join(' ')

    #md['soup'].fillna({})
    #print(md['soup'])

    count = CountVectorizer(analyzer='word',ngram_range=(1,1),min_df=0, stop_words='english')
    count_matrix = count.fit_transform(md['soup'])
    #print (count_matrix.shape)
    #print np.array(count.get_feature_names())
    #print(count_matrix.shape)

    cosine_sim = cosine_similarity(count_matrix, count_matrix)


    # In[91]:


    def build_user_profiles():
        user_profiles=np.zeros((53421,999))
        #print(rd.iloc[0]['user_id'])
	#len(rd['book_id'])
        for i in range(0,1000):
            u=rd.iloc[i]['user_id']
            b=rd.iloc[i]['book_id']
            #print(u,b)
            #print(i)
            #if b<999:
                #print("match at "+str(b))
            user_profiles[u][b-1]=rd.iloc[i]['rating']
        #print(user_profiles)
        return user_profiles

    user_profiles=build_user_profiles()
    def _get_similar_items_to_user_profile(person_id):
            #Computes the cosine similarity between the user profile and all item profiles
            #print(user_profiles[person_id])
        #print("\n---------\n")
        #print(cosine_sim[0])
        user_ratings = np.empty((999,1))
        cnt=0
        for i in range(0,998):
            book_sim=cosine_sim[i]
            user_sim=user_profiles[person_id]
            user_ratings[i]=(book_sim.dot(user_sim))/sum(cosine_sim[i])
        maxval = max(user_ratings)
    #print(maxval)

        for i in range(0,998):
            user_ratings[i]=((user_ratings[i]*5.0)/(maxval))
            #print(user_ratings[i])
            if(user_ratings[i]>3):
                #print("MILA KUCCHHH")
                cnt+=1
        #print(max(user_ratings))
        #print (cnt)
       
            #print(cosine_similarities)
            
            #return similar_items
        return user_ratings
    content_ratings = _get_similar_items_to_user_profile(userId)



    # In[100]:


    num = md[['book_id']]
    #print(num)

    num1 = pd.DataFrame(data=content_ratings[0:,0:])


    frames = [num, num1]
    #result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])

    mer = pd.concat(frames, axis =1,join_axes=[num.index])
    mer.columns=['book_id', 'content_rating']
    #print(mer.shape)
    #print('here')
    #print(mer)





    # In[102]:


    ## for user 2 #

#print(temp_ratings.shape)
    cb = temp_ratings[(temp_ratings['user_id'] == userId)][['book_id', 'rating']]
#   print(cb.shape)
#   print(pop.shape)
    hyb = md[['book_id']]
    hyb = hyb.merge(cb,on = 'book_id')
    hyb = hyb.merge(pop, on='book_id')
    hyb = hyb.merge(mer, on='book_id')
    #hyb.shape


    # In[106]:


    def weighted_rating(x):
        v = x['rating']
        R = x['popularity_rating']
        c = x['content_rating']
        return 0.4*v + 0.2*R + 0.4 * c


    # In[107]:


    print(hyb)
    hyb['final'] = hyb.apply(weighted_rating, axis=1)
    hyb = hyb.sort_values('final', ascending=False).head(999)
    #print(hyb['final'])

    print(hyb)
    return hyb
    #print(len(hyb['final']))

def user_profile(userId,train_rd):
        user_profile=np.zeros((1,999))
        for i in range(0,200000):
		u=rd.iloc[i]['user_id']
         	b=rd.iloc[i]['book_id']
		if u==userId:
			user_profile[0][b-1]=rd.iloc[i]['rating']
            
	print(user_profile)
	return user_profile


rd = pd.read_csv('CustomData/FinalRatings.csv')

train_rd, test_rd = train_test_split(rd,
                                   test_size=0.20,
                                   random_state=42)

users_selected=[1]
avg_rmse=0
for i in users_selected:

	ms=0
	cnt=0
	user_profile=user_profile(i,train_rd)
	#predicted = hybrid(i,train_rd)
	for x in range(0,998):
		u=user_profile[0][x]
		if u>0:
			print("MILA")			
			#y=predicted[(predicted['book_id']==x)][['final']]
			ms+=(0-u)*(0-u)
			cnt+=1
	rmse=math.sqrt(ms/cnt)
	avg_rmse+=rmse
avg_rmse/=3

print("\n---------\n")
print("Avg RMSE: "+ str(avg_rmse))
print("\n---------\n")
	
	
