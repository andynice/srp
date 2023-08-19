## WORDCLOUD
from os import path
import os
from wordcloud import WordCloud
#
# #create data
most_freq_words_3 = sorted(freq_count, key=freq_count.get, reverse=True)[:3]
most_freq_words_25 = sorted(freq_count, key=freq_count.get, reverse=True)[:25]
selected_words = ['covid', 'covid-19', 'coronavirus', 'vaccine', 'positive', 'corona', 'pandemic', 'virus', 'health', 'death', 'died', 'case', 'test']
freq_count_subset = dict(filter(lambda i:i[0] in most_freq_words_3, freq_count.items()))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=50, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate_from_frequencies(freq_count)
# Visualize the word cloud
#wordcloud.to_image()
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
wordcloud.to_file(path.join(d, "./output/visualizations/wordcloud.png"))


## MIXED LINE GRAPHS
# importing package
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates

freqs = []

for word in most_freq_words_3:
    freqs.append(np.random.randint(1,5000,58))

date_range = ['2021-01-01', '2021-02-28']

start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
dates_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

# plot lines
for idx, word in enumerate(most_freq_words_3):
    plt.xlabel("Days")
    plt.ylabel("Frequencies")
    plt.title("Frequencies of word '" + word + "' VS. Days"
              "\n during January and February 2021")
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate()  # Rotation
    
    plt.plot(dates_generated, freqs[idx])
    plt.savefig('./output/visualizations/' + word + '-line-graph.png')

    plt.show()



## BAR GRAPHS
freqs = [freq_count[word] for word in most_freq_words_25]
plt.bar(most_freq_words_25, freqs)
plt.xlabel("Words")
plt.ylabel("Frequencies")
plt.title("Frequencies of 25 most frequent words"
          "\n during January and February 2021")

plt.xticks(rotation=90)
plt.savefig('./output/visualizations/bar-graph-n.png', bbox_inches='tight')
plt.show()

selected_words_indices = []
i = 0
for word in most_freq_words_25:
    if word in selected_words:
        selected_words_indices.append(i)
    i += 1

barlist = plt.bar(most_freq_words_25, freqs)
for idx in selected_words_indices:
    barlist[idx].set_color('orange')
plt.xlabel("Words")
plt.ylabel("Frequencies")
plt.title("Frequencies of 25 most frequent words"
          "\n during January and February 2021"
          "\n with COVID words highlighted")

plt.xticks(rotation=90)
plt.savefig('./output/visualizations/bar-graph-m.png', bbox_inches='tight')
plt.show()
