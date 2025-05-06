import tweepy
from textblob import TextBlob
import os
from dotenv import load_dotenv
import re

# Load environment variables for API keys
load_dotenv()

# --- Twitter API Credentials (Replace with your actual keys or load from .env) ---
# It's highly recommended to store these securely in a .env file
# and add .env to your .gitignore
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "YOUR_API_KEY")
TWITTER_API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY", "YOUR_API_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "YOUR_ACCESS_TOKEN_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YOUR_BEARER_TOKEN") # Needed for v2 API

class SentimentAnalyzer:
    def __init__(self):
        """Initialize Twitter API v2 client."""
        if not all([TWITTER_BEARER_TOKEN, TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET]):
            print("Warning: Twitter API credentials not fully configured. Sentiment analysis might fail.")
            print("Please set TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, and TWITTER_BEARER_TOKEN environment variables.")
            self.client = None
        else:
            # Use v2 API client
            self.client = tweepy.Client(
                bearer_token=TWITTER_BEARER_TOKEN,
                consumer_key=TWITTER_API_KEY,
                consumer_secret=TWITTER_API_SECRET_KEY,
                access_token=TWITTER_ACCESS_TOKEN,
                access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
            )
            print("Twitter API v2 Client Initialized.")

    def _clean_tweet(self, tweet_text):
        """Utility function to clean tweet text by removing links, special characters using simple regex.
        Args:
            tweet_text (str): The raw text of the tweet.
        Returns:
            str: Cleaned tweet text.
        """
        # Remove URLs
        cleaned = re.sub(r'http\S+|www\S+|https\S+', '', tweet_text, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        cleaned = re.sub(r'\@\w+|#','', cleaned)
        # Remove punctuation and special characters (optional, depending on analysis needs)
        # cleaned = re.sub(r'[^A-Za-z0-9\s]+', '', cleaned)
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def get_tweet_sentiment(self, tweet_text):
        """Utility function to classify sentiment of passed tweet using textblob's sentiment method.
        Args:
            tweet_text (str): The text of the tweet.
        Returns:
            float: Polarity score (-1.0 to 1.0).
        """
        cleaned_text = self._clean_tweet(tweet_text)
        analysis = TextBlob(cleaned_text)
        # Polarity score ranges from -1 (negative) to 1 (positive)
        return analysis.sentiment.polarity

    def fetch_tweets_and_analyze(self, search_query_string, max_tweets=10):
        """Fetches recent tweets based on a search query and calculates the average sentiment.
        Args:
            search_query_string (str): The search query string (e.g., 'Bitcoin lang:en -is:retweet').
            max_tweets (int): Maximum number of newest tweets to fetch. Min 10, Max 100 for recent search.
        Returns:
            float: Average sentiment polarity score, or 0.0 if fetching fails or no tweets found.
        """
        if not self.client:
            print("Twitter client not initialized. Cannot fetch tweets.")
            return 0.0

        all_sentiments = []
        total_tweets_processed = 0
        total_tweets_fetched = 0

        print(f"\nFetching tweets with query: '{search_query_string}'")
        try:
            # Ensure max_tweets is within the API's valid range [10, 100] for search_recent_tweets
            actual_max_results = max(10, min(max_tweets, 100))
            
            response = self.client.search_recent_tweets(
                query=search_query_string,
                max_results=actual_max_results, 
                tweet_fields=["created_at", "public_metrics", "author_id"]
            )

            if response.data:
                print(f"Fetched {len(response.data)} tweets for query: '{search_query_string}'")
                total_tweets_fetched += len(response.data)
                processed_for_query = 0
                for tweet in response.data:
                    # Optional: Detailed logging for each tweet can be kept or removed based on verbosity needs
                    # print(f"--- Tweet ID: {tweet.id} ---")
                    # if tweet.public_metrics:
                    #     print(f"  Likes: {tweet.public_metrics.get('like_count', 'N/A')}, Retweets: {tweet.public_metrics.get('retweet_count', 'N/A')}")
                    # print(f"  Text: {tweet.text[:100]}...") # Print first 100 chars
                    # print("--------------------------------------------------")

                    sentiment = self.get_tweet_sentiment(tweet.text)
                    all_sentiments.append(sentiment)
                    processed_for_query += 1
                
                if processed_for_query > 0:
                    total_tweets_processed += processed_for_query
                    print(f"Processed {processed_for_query} tweets for query '{search_query_string}'.")
                else:
                    print(f"No tweets processed for query '{search_query_string}'.")
            else:
                print(f"No recent tweets found for query: '{search_query_string}'")

        except tweepy.errors.TweepyException as e:
            print(f"Error fetching tweets for query '{search_query_string}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while fetching for query '{search_query_string}': {e}")

        if not all_sentiments:
            print("No sentiments collected from the query.")
            return 0.0

        average_sentiment = sum(all_sentiments) / len(all_sentiments)
        print(f"\nOverall average sentiment from {total_tweets_processed} tweets (out of {total_tweets_fetched} fetched): {average_sentiment:.4f}")
        return average_sentiment

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    if analyzer.client:
        # Example of fetching with a general query
        general_search_query = "Bitcoin lang:en -is:retweet"
        # Note: max_tweets will be clamped to [10,100] by the function
        sentiment_score = analyzer.fetch_tweets_and_analyze(search_query_string=general_search_query, max_tweets=15) 
        print(f"\nFinal average sentiment for query '{general_search_query}': {sentiment_score:.4f}")

        # Example with a different query and max_tweets
        another_query = "Ethereum OR #ETH lang:en -is:retweet -is:reply"
        sentiment_score_another = analyzer.fetch_tweets_and_analyze(search_query_string=another_query, max_tweets=20)
        print(f"\nFinal average sentiment for query '{another_query}': {sentiment_score_another:.4f}")
    else:
        print("Could not run example due to missing Twitter API credentials.")