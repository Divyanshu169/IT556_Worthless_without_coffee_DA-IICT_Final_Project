import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
from collections import defaultdict


class hybrid(object):
    
    def __init__ (self,user_id,ratings):
        
        self.user_id = user_id
        self.md = pd.read_csv('CustomData/FinalData.csv')
        self.ratings = ratings
        print(ratings[(ratings['user_id'] == user_id)][['user_id','book_id', 'rating']])
	
        self.popularity_rating = self.popularity(self.md)
        self.collaborative_rating = self.collaborative(self.ratings, self.user_id)
        self.content_rating = self.content_based(self.md,self.ratings,self.user_id)
        self.final_hybrid(self.md, self.popularity_rating , self.collaborative_rating, self.content_rating, self.user_id)
        


#Popularity#

    def popularity(self,md):
		
        fd = pd.read_csv('CustomData/AverageRatings.csv')
        fd1 = pd.read_csv('CustomData/RatingsCount.csv')
	
        fd[fd['rating'].notnull()]['rating'] = fd[fd['rating'].notnull()]['rating'].astype('float')
        vote_averages= fd[fd['rating'].notnull()]['rating'] 
        C = vote_averages.mean()

        fd1[fd1['rating'].notnull()]['rating'] = fd1[fd1['rating'].notnull()]['rating'].astype('float')
        vote_counts = fd1[fd1['rating'].notnull()]['rating']
        m = len(vote_counts)

        md['ratings_count'] = fd1['rating']
        md['average_rating'] = fd['rating']

        qualified = md[(md['ratings_count'].notnull())][['book_id','title', 'authors', 'ratings_count', 'average_rating']]

        qualified['ratings_count'] = qualified['ratings_count'].astype('float')

        qualified['average_rating'] = qualified['average_rating'].astype('float')

        qualified.shape

        def weighted_rating(x):
            v = x['ratings_count']
            R = x['average_rating']
            return (v/(v+m) * R) + (m/(m+v) * C)

        qualified['popularity_rating'] = qualified.apply(weighted_rating, axis=1)
        pop = qualified[['book_id','popularity_rating']]
        print(qualified.shape)
        print(pop.shape)

        return pop
    ### Collaborative ##

    def collaborative(self,ratings,user_id):

        reader = Reader()
        #ratings.head()

        temp_ratings = ratings
        data = Dataset.load_from_df(temp_ratings[['user_id', 'book_id', 'rating']], reader)
        data.split(n_folds=2)

        ## Training the data ##
        svd = SVD()
        evaluate(svd, data, measures=['RMSE', 'MAE'])

        trainset = data.build_full_trainset()

        algo = SVD()
        algo.fit(trainset)

        #svd.train(trainset)
        ## Testing the data ##
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        count = 0
     
        for uid, iid, true_r, est, _ in predictions:
             if uid == user_id:
                count = count+1
                temp_ratings.loc[len(temp_ratings)+1]= [uid,iid,est]

        cb = temp_ratings[(temp_ratings['user_id'] == user_id)][['book_id', 'rating']]

        return(cb)


    ##### CONTENT ######

    def content_based(self,md,ratings,user_id):       

        md['book_id'] = md['book_id'].astype('int')
        ratings['book_id'] = ratings['book_id'].astype('int')
        ratings['user_id'] = ratings['user_id'].astype('int')
        ratings['rating'] = ratings['rating'].astype('int')
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

        count = CountVectorizer(analyzer='word',ngram_range=(1,1),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(md['soup'])
        print (count_matrix.shape)

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        def build_user_profiles():
            user_profiles=np.zeros((60001,999))
		#taking only the first 100000 ratings to build user_profile
            for i in range(0,100000):
                u=ratings.iloc[i]['user_id']
                b=ratings.iloc[i]['book_id']
                user_profiles[u][b-1]=ratings.iloc[i]['rating']   
            return user_profiles

        user_profiles=build_user_profiles()

        def _get_similar_items_to_user_profile(person_id):
            #Computes the cosine similarity between the user profile and all item profiles

            user_ratings = np.empty((999,1))
            cnt=0
            for i in range(0,998):
                book_sim=cosine_sim[i]
                user_sim=user_profiles[person_id]
                user_ratings[i]=(book_sim.dot(user_sim))/sum(cosine_sim[i])
            maxval = max(user_ratings)
            print(maxval)

            for i in range(0,998):
                user_ratings[i]=((user_ratings[i]*5.0)/(maxval))
                if(user_ratings[i]>3):
                    cnt+=1

            return user_ratings

        content_ratings = _get_similar_items_to_user_profile(user_id)
	
        num = md[['book_id']]
        num1 = pd.DataFrame(data=content_ratings[0:,0:])
        frames = [num, num1]


        content_rating = pd.concat(frames, axis =1,join_axes=[num.index])
        content_rating.columns=['book_id', 'content_rating']

        return(content_rating)

    
    def final_hybrid(self,md, popularity_rating , collaborative_rating, content_rating, user_id):

        hyb = md[['book_id']]
        title = md[['book_id','title', 'Genres']]

        hyb = hyb.merge(title,on = 'book_id')
        hyb = hyb.merge(self.collaborative_rating,on = 'book_id')
        hyb = hyb.merge(self.popularity_rating, on='book_id')
        hyb = hyb.merge(self.content_rating, on='book_id')

        def weighted_rating(x):
            v = x['rating']
            R = x['popularity_rating']
            c = x['content_rating']
            return 0.4*v + 0.2*R + 0.4 * c

        hyb['hyb_rating'] = hyb.apply(weighted_rating, axis=1)
        hyb = hyb.sort_values('hyb_rating', ascending=False).head(999)
        hyb.columns = ['Book ID' , 'Title', 'Genres', 'Collaborative Rating', 'Popularity Rating' , 'Content Rating', 'Hybrid Rating']

        print(len(hyb['Hybrid Rating']))
        print(hyb)


def newUser():
    print('\n Rate from books\n')
    print('ID   Author                           Title                                                              Genre\n')
    print('2.   J.K. Rowling, Mary               Harry Potter and the Sorcerer\'s Stone (Harry Potter, #1)      Fantasy;Young-Age')
    print('127. Malcolm Gladwell                 The Tipping Point: How Little Things Can Make a Big Difference Self-Help')
    print('239. Max Brooks                       World War Z: An Oral History of the Zombie War                 Horror;Fiction')
    print('26   Dan Brown                        The Da Vinci Code                                              Thriller;Drama')
    print('84   Michael Crichton                 Jurassic Park (Jurassic Park, #1)                              SciFi;Thriller;Fantasy')
    print('86   John Grisham                     A Time to Kill                                                 Thriller')
    print('966  Scott Turow                      Presumed Innocent                                              Thriller;Crime')
    print('42   Louisa May Alcott                Little Women (Little Women, #1)                                Young-Age;Romance;Drama')
    print('44   Nicholas Sparks                  The Notebook (The Notebook, #1)                                Romance;Drama')
    print('54   Douglas Adams                    The Hitchhiker\'s Guide to the Galaxy                          Fantasy;Fiction')
    print('134  Cassandra Clare                  City of Glass (The Mortal Instruments, #3)                     Kids;Fantasy;Fiction')
    print('399  J.K. Rowling                     The Tales of Beedle the Bard                                               Kids;Fantasy;Fiction')
    print('38   Audrey Niffenegger               The Time Traveler\'s Wife                                                  Romance;SciFi;Fantasy;Domestic')
    print('729  Dan Simmons                      Hyperion (Hyperion Cantos, #1)                                             SciFi')
    print('807  Dave Eggers                      The Circle                                                                 SciFi')
    print('690  Barack Obama                     The Audacity of Hope: Thoughts on Reclaiming the American Dream            Biography')
    print('617  Piper Kerman                     Orange Is the New Black                                                    Biography')
    print('495  Dave Eggers                      A Heartbreaking Work of Staggering Genius                                  Biography')
    print('770  William Shakespeare,Roma Gill    Julius Caesar                                                              History;Classic')
    print('773  William Shakespeare              The Taming of the Shrew                                                    Comedy;Classic')
    print('829  E.M. Forster                     A Room with a View                                                         Classic')
    print('971  Marcus Pfister, J. Alison James  The Rainbow Fish                                                           Kids')
    print('976  Robert Kapilow, Dr. Seuss        Dr. Seuss\'s Green Eggs and Ham: For Soprano, Boy Soprano, and Orchestra   Kids')
    print('627  Jon Scieszka, Lane Smith         The True Story of the 3 Little Pigs                                        Kids;Fiction')
    print('121  Vladimir Nabokov, Craig Raine    Lolita                                                                     Biography;Romance;Comedy')
    print('196  Chuck Palahniuk                  Fight Club                                                                 Comedy;Drama')
    print('444  A.A. Milne, Ernest H. Shepard    Winnie-the-Pooh (Winnie-the-Pooh, #1)                                      Kids;Comedy')
    print('745  Jenny  Lawson                    Lets Pretend This Never Happened: A Mostly True Memoir                     Biography;Comedy')


    
    ratings = pd.read_csv('CustomData/FinalRatings.csv')
   
    #taking only the first 100000 ratings
    ratings=ratings[1:100000]
    
    user_id = 60000
    rating_count = len(ratings['user_id'])+1
    
    print(user_id)
    print('\n----------------Welcome User '+str(user_id)+'-------------------')
    print('\nPlease Rate 5 books from the above list.')
    
    for x in range(0,5):
        print("\n")
        bookId=input("BookId:")
       
        rating=input("Rating:")
        
        ratings.loc[rating_count]= [user_id,bookId,rating]
        rating_count =rating_count+1
        
    h = hybrid(user_id,ratings)
        

print("------------------------------Welcome to the Book Recommendation Engine---------------------------\n")

user=raw_input("1. Book Recommendation for New User. \n2. Book Recommendation for Existing User.\n")

if user=='1':
    newUser()
    
elif user=='2':
    ratings = pd.read_csv('CustomData/FinalRatings.csv')
    ratings=ratings[1:100000]
    #taking only the first 100000 ratings
    userId=int(raw_input("\nPlease Enter User Id: "))
    print('\n----------------Welcome User'+str(userId)+'-------------------')
    h = hybrid(userId,ratings)
    
else:
    print("Invalid option\n ")


