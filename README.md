# IT556_Worthless_without_coffee_DA-IICT_Final_Project
This is a book recommendation engine built using a hybrid model of Collaborative filtering, Content Based Filtering and Popularity Matrix for our course IT556 - Recommendation Engines.

The Team
1. Divyanshu Shekhar
2. Sakshi Sharma
3. Keya Desai

## Dataset
Dataset used for the project is goodbooks-10k. It contains contains six million ratings for ten thousand most popular (with most ratings) books. However, the dataset is missing genre tags for each book. For content based filtering, genre of the book is important and hence we have tagged the first thousand books manually. <br/><br /> The genres included are: 'Fantasy', 'Fiction', 'Self-Help', 'Drama', 'Romance', 'Thriller', 'Biography', 'Erotic', 'Kids', 'Poetry', 'Horror', 'History', 'Academic', 'Comedy', 'Classic', ' Thriller', 'Domestic Fiction’, ‘SciFi’, ‘Crime Fiction’, ‘Psychological Fiction’, ‘Young-Adult Fiction’. <br /><br /> The final data hence contains book-id, authors, title and genres for thousand books.<br />

## Model

### Popularity
The basic idea behind this recommender is that movies that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience. This model does not give personalized recommendations based on the user.From the ratings matrix, average ratings and rating count for each book is calculated. Then, IMDB's weighted rating formula is used to construct a chart. Mathematically, it is represented as follows:<br /><br />Weighted Rating (WR) =  (v/(v+m) * R) + (m/(v+m) * C)<br />where,<br /><br />v is the number of votes for the movie<br />m is the minimum votes required to be listed in the chart<br />R is the average rating of the movie<br />C is the mean vote across the whole report<br />

### Content Based Filtering 
The engine above is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a book will receive the same recommendations for that book, regardless of who s/he is.<br /><br />To personalise our recommendations more, we have build an engine that computes similarity between books based on certain metrics and suggests books that are most similar to a particular book that a user liked. <br /><br />The Content based Recommenders are built using:<br />1. Authors <br />2. Genres<br /><br />We have given twice the weigth to author of the book. Cosine Similarity is used to calculate a numeric quantity that denotes the similarity between two books. Mathematically, it is defined as follows:<br />cosine(x,y) = x.y⊺/ (||x||.||y||)<br />

### Collaborative Based Filtering
The content based engine is only capable of suggesting books which are close to a certain book. That is, it is not capable of capturing tastes and providing recommendations across genres.<br /><br/> Therefore, we have used Collaborative Filtering to make recommendations. Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.<br /><br />Surprise library has been used that uses extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give great recommendations. We get a RMSE of 0.8840.

### Hybrid

We combine the ratings from Popularity model, Conten based filtering and Collaborative filtering to get more accurate results.It gives the predicted rating as weighted combination of the above described methods. Equal weigths have been given to collaborative and content rating.<br /><br /> R<sub>hybrid</sub> = (1-2α)* R<sub>popularity</sub> + α* R<sub>collaborative</sub>  + α* R<sub>content</sub> <br /><br /> α = 0.4

### Training and Testing
For training and testing, the data was split in the ratio of 80:20. 80% for training and 20% for testing. We have used the rmse method to calculate accuracy of the predicted ratings by our recommendation system with respect to the actual rating given by user to a particular book. For the hybrid model, we get RMSE of 0.6960 which is considerably better than any other method.


## Try it out
final.py can directly be run to start the recommendation engine. We have made a console for the same. Book recommendations can be obtained for an existing user or a new user. New user to asked to provide ratings for any 5 books from the list of books that has been generated using popularity ratings, genre wise. Based on these ratings, books are recommended to the user. 

## Future Extension
Given more time, we would have used Stochastic Gradient Descent to learn the weights given to each of Popularity rating, Content based rating and collaborative rating. 
