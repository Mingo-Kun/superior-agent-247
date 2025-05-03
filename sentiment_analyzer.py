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

    def fetch_tweets_and_analyze(self, query, max_results=10):
        """Fetches recent tweets based on a query and calculates the average sentiment.
        Args:
            query (str): The search query (e.g., '$BTC', 'Bitcoin price').
            max_results (int): Maximum number of tweets to fetch (min 10, max 100 for recent search).
        Returns:
            float: Average sentiment polarity score, or 0.0 if fetching fails or no tweets found.
        """
        if not self.client:
            print("Twitter client not initialized. Cannot fetch tweets.")
            return 0.0

        sentiments = []
        try:
            # Use search_recent_tweets from v2 API
            # Note: Requires Elevated access for v2 search endpoints usually.
            # Standard Essential access might have limitations.
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max(10, min(max_results, 100)), # Ensure max_results is within API limits (10-100)
                tweet_fields=["created_at", "public_metrics"]
            )

            if response.data:
                print(f"Fetched {len(response.data)} tweets for query: '{query}'")
                for tweet in response.data:
                    sentiment = self.get_tweet_sentiment(tweet.text)
                    sentiments.append(sentiment)
                    # print(f"Tweet: {tweet.text}\nSentiment: {sentiment:.2f}\n---")
            else:
                print(f"No recent tweets found for query: '{query}'")
                return 0.0

        except tweepy.errors.TweepyException as e:
            print(f"Error fetching tweets: {e}")
            return 0.0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return 0.0

        if not sentiments:
            return 0.0

        average_sentiment = sum(sentiments) / len(sentiments)
        print(f"Average sentiment for '{query}': {average_sentiment:.4f}")
        return average_sentiment

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    if analyzer.client:
        # Example: Analyze sentiment for Bitcoin
        btc_query = 'Bitcoin OR $BTC -is:retweet lang:en'
        avg_sentiment = analyzer.fetch_tweets_and_analyze(btc_query, max_results=20)
        print(f"\nOverall average sentiment for '{btc_query}': {avg_sentiment:.4f}")
    else:
        print("Could not run example due to missing Twitter API credentials.")