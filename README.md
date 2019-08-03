# twitter-sentiment-analysis
A sentiment analysis for live twitter tweets based on the self trained voting based algorithm.

## Steps to Run the Code:
1.  Install the required python packages using "requirements.txt" file. If there is a problem with "pip install requirements.txt" then try the command "pip install --upgrade -r requirements.txt" instead.
2. Create a folder in the root directory named "pickled_algos". Due to huge size of the pickle files, it could not be uploaded to github. 
3. Create an app for twitter API to get access to the consumer and access token keys. Replace the keys with yours in the file "twitter_analysis.py". You can save the keys inside your OS environment variables with the same name present in the code. 
4.  Run the file "twitter_analysis.py". Running for the first time will take some time since it does training for 7 different scikitlearn algorithms. It then saves the models as pickle files inside the "pickled_algos" folder. After the first time, code execution time will improve.
4. You can specify your own topic to stream and perform sentiment analysis on the topic by changing the string inside "twitterStream.filter(track=["obama"])" command. 
5. Run the file "live_visualization.py" to see the trend of positive and negative tweets over the streaming period. It shows live graph while the file inside "output/twitter-output.txt" is being updated.
