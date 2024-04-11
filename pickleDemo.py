import pandas as pd
import numpy as np
import joblib

class SentimentRecommender:
    root_model_path = "models/"
    sentiment_model_path = "lr_base_model.pkl"
    tfidf_vectorizer_path = "tfidf.pkl"
    best_recommender_path = "best_recommendation_model.pkl"
    clean_dataframe_path = "clean_data.pkl"

    def __init__(self):
        self.sentiment_model = joblib.load(self.root_model_path + self.sentiment_model_path)
        self.tfidf_vectorizer = joblib.load(self.root_model_path + self.tfidf_vectorizer_path)
        self.user_final_rating = joblib.load(self.root_model_path + self.best_recommender_path)
        self.cleaned_data = joblib.load(self.root_model_path + self.clean_dataframe_path)
    
    def top5_recommendations(self, user_name):
            if user_name not in self.user_final_rating.index:
                print(f"The User {user_name} does not exist. Please provide a valid user name")
                return None
            else:
                # Get top 20 recommended products from the best recommendation model
                top20_recommended_products = list(self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
                
                # Get only the recommended products from the prepared dataframe "df_sent"
                df_top20_products = self.cleaned_data[self.cleaned_data.id.isin(top20_recommended_products)]
                
                # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
                X = self.tfidf_vectorizer.transform(df_top20_products["reviews_lemmatized"].values.astype(str))
                
                # Use the best sentiment model to predict the sentiment for these user reviews
                df_top20_products['predicted_sentiment'] = self.sentiment_model.predict(X)
                
                # Create a new column to map Positive sentiment to 1 and Negative sentiment to 0. This will allow us to easily summarize the data
                df_top20_products['positive_sentiment'] = df_top20_products['predicted_sentiment'].apply(lambda x: 1 if x == "Positive" else 0)
                
                # Create a new dataframe "pred_df" to store the count of positive user sentiments
                pred_df = df_top20_products.groupby(by='name').sum()
                
                 # Create a new dataframe "pred_df" to store the count of positive user sentiments
                pred_df = df_top20_products.groupby(by='name').agg({'positive_sentiment': 'sum', 'predicted_sentiment': 'count'})
        
                 # Rename columns
                pred_df.columns = ['pos_sent_count', 'total_sent_count']
        
                # Create a column that measures the % of positive user sentiment for each product review
                pred_df['post_sent_percentage'] = np.round(pred_df['pos_sent_count'] / pred_df['total_sent_count'] * 100, 2)
        
                # Return top 5 recommended products to the user
                result = pred_df.sort_values(by='post_sent_percentage', ascending=False).head(5)
                return result
