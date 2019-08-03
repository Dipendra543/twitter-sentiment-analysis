from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import os

import text_classification as tc

# consumer key, consumer secret, access token, access secret.

consumer_key = os.getenv("twitter_consumer_key")
consumer_secret = os.getenv("twitter_consumer_secret")
access_token = os.getenv("twitter_access_token")
access_secret = os.getenv("twitter_access_secret")


class Listener(StreamListener):
    def on_status(self, status):
        print("Tweet Arrived")
        print(status)

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = tc.find_sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence * 100 >= 80:
            output = open("output/twitter-out.txt", "a")  # Open the file in append mode
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        # username = all_data["user"]["screen_name"]
        # print(username, tweet)
        return True

    def on_connect(self):
        print("Successfully connected to streaming server")


    def on_error(self, status):
        # print(self.api)
        print("Error Status: ", status)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["obama"])

