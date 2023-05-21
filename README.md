# srp
SRP Project in Hildesheim university. Prediction Model based on tweets to predict covid-19 cases


## Stopwords
https://countwordsfree.com/stopwords/
len(stop_words_english):851
len(stop_words_german):592
nltk
len(stop_words_english):179
len(stop_words_german):232



## Install
### demoji
pip install demoji

### spacy
python -m spacy download de

### WordCloud
https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
conda install -c conda-forge wordcloud

If after this installation if throws errors like
DLL load failed while importing _imaging: The specified module could not be found

conda uninstall pillow --force-remove
pip install pillow