#conda install -c conda-forge wordcloud
from os import path
import os
from wordcloud import WordCloud

clean_tweet = "test1 test2 test3 test5 test1"
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=50, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(clean_tweet)
# Visualize the word cloud
#wordcloud.to_image()
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
wordcloud.to_file(path.join(d, "wordcloud.png"))