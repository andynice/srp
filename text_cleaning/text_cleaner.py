import pandas as pd
import re
import demoji
# Natural Language Toolkit
import nltk
# Must be executed on the first run
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import spacy
import datetime
from time import time

## READ AND PROCESS DATAFRAME
def read_date_file(date):
    filename = "./data/" + date + ".csv"
    df_data = pd.read_csv(filename)
    return df_data

def filter_dataframe(df_data, langs):

    filtered_df = df_data[df_data["lang"].isin(langs)]

    #cols_to_drop = ['geo', 'id', 'source']
    cols_to_drop = ['geo', 'id', 'source', 'author id', 'like_count', 'quote_count', 'reply_count', 'retweet_count']
    filtered_df = df_data.drop(cols_to_drop, axis=1, errors="ignore")

    # Convert "created_at" column to datetime
    filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'])

    # Remove the minutes, seconds, and timezone offset
    # data['created_at'] = data['created_at'].dt.strftime('%Y-%m-%d %H:00:00')
    filtered_df['created_at'] = filtered_df['created_at'].dt.strftime('%Y-%m-%d')

    return [filtered_df[filtered_df["lang"] == lang] for lang in langs]

## CLEAN TEXT
en_stopwords = ["able","about","above","abroad","according","accordingly","across","actually","adj","after","afterwards","again","against","ago","ahead","ain't","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","a's","aside","ask","asking","associated","at","available","away","awfully","back","backward","backwards","be","became","because","become","becomes","becoming","been","before","beforehand","begin","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","came","can","cannot","cant","can't","caption","cause","causes","certain","certainly","changes","clearly","c'mon","co","co.","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","c's","currently","dare","daren't","definitely","described","despite","did","didn't","different","directly","do","does","doesn't","doing","done","don't","down","downwards","during","each","edu","eg","eight","eighty","either","else","elsewhere","end","ending","enough","entirely","especially","et","etc","even","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","fairly","far","farther","few","fewer","fifth","first","five","followed","following","follows","for","forever","former","formerly","forth","forward","found","four","from","further","furthermore","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","had","hadn't","half","happens","hardly","has","hasn't","have","haven't","having","he","he'd","he'll","hello","help","hence","her","here","hereafter","hereby","herein","here's","hereupon","hers","herself","he's","hi","him","himself","his","hither","hopefully","how","howbeit","however","hundred","i'd","ie","if","ignored","i'll","i'm","immediate","in","inasmuch","inc","inc.","indeed","indicate","indicated","indicates","inner","inside","insofar","instead","into","inward","is","isn't","it","it'd","it'll","its","it's","itself","i've","just","k","keep","keeps","kept","know","known","knows","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","likewise","little","look","looking","looks","low","lower","ltd","made","mainly","make","makes","many","may","maybe","mayn't","me","mean","meantime","meanwhile","merely","might","mightn't","mine","minus","miss","more","moreover","most","mostly","mr","mrs","much","must","mustn't","my","myself","name","namely","nd","near","nearly","necessary","need","needn't","needs","neither","never","neverf","neverless","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","no-one","nor","normally","not","nothing","notwithstanding","novel","now","nowhere","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","one's","only","onto","opposite","or","other","others","otherwise","ought","oughtn't","our","ours","ourselves","out","outside","over","overall","own","particular","particularly","past","per","perhaps","placed","please","plus","possible","presumably","probably","provided","provides","que","quite","qv","rather","rd","re","really","reasonably","recent","recently","regarding","regardless","regards","relatively","respectively","right","round","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","shan't","she","she'd","she'll","she's","should","shouldn't","since","six","so","some","somebody","someday","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","that'll","thats","that's","that've","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","there'd","therefore","therein","there'll","there're","theres","there's","thereupon","there've","these","they","they'd","they'll","they're","they've","thing","things","think","third","thirty","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","till","to","together","too","took","toward","towards","tried","tries","truly","try","trying","t's","twice","two","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","upwards","us","use","used","useful","uses","using","usually","v","value","various","versus","very","via","viz","vs","want","wants","was","wasn't","way","we","we'd","welcome","well","we'll","went","were","we're","weren't","we've","what","whatever","what'll","what's","what've","when","whence","whenever","where","whereafter","whereas","whereby","wherein","where's","whereupon","wherever","whether","which","whichever","while","whilst","whither","who","who'd","whoever","whole","who'll","whom","whomever","who's","whose","why","will","willing","wish","with","within","without","wonder","won't","would","wouldn't","yes","yet","you","you'd","you'll","your","you're","yours","yourself","yourselves","you've","zero","a","how's","i","when's","why's","b","c","d","e","f","g","h","j","l","m","n","o","p","q","r","s","t","u","uucp","w","x","y","z","I","www","amount","bill","bottom","call","computer","con","couldnt","cry","de","describe","detail","due","eleven","empty","fifteen","fifty","fill","find","fire","forty","front","full","give","hasnt","herse","himse","interest","itse”","mill","move","myse”","part","put","show","side","sincere","sixty","system","ten","thick","thin","top","twelve","twenty","abst","accordance","act","added","adopted","affected","affecting","affects","ah","announce","anymore","apparently","approximately","aren","arent","arise","auth","beginning","beginnings","begins","biol","briefly","ca","date","ed","effect","et-al","ff","fix","gave","giving","heres","hes","hid","home","id","im","immediately","importance","important","index","information","invention","itd","keys","kg","km","largely","lets","line","'ll","means","mg","million","ml","mug","na","nay","necessarily","nos","noted","obtain","obtained","omitted","ord","owing","page","pages","poorly","possibly","potentially","pp","predominantly","present","previously","primarily","promptly","proud","quickly","ran","readily","ref","refs","related","research","resulted","resulting","results","run","sec","section","shed","shes","showed","shown","showns","shows","significant","significantly","similar","similarly","slightly","somethan","specifically","state","states","stop","strongly","substantially","successfully","sufficiently","suggest","thered","thereof","therere","thereto","theyd","theyre","thou","thoughh","thousand","throug","til","tip","ts","ups","usefully","usefulness","'ve","vol","vols","wed","whats","wheres","whim","whod","whos","widely","words","world","youd","youre"]
# remove words
# research word is part of non-stop words in english
en_stopwords.remove("research")
# add words
# dont word is not part of non-stop words in english
# en_stopwords.append("dont")
# en_stopwords.append("wouldnt")
# en_stopwords.append("ill")
# en_stopwords.append("itll")
# en_stopwords.append("weve")
# en_stopwords.append("ive")
en_stopwords.append("would've")
en_stopwords.append("wouldve")
en_stopwords.append("y'all")
en_stopwords.append("y'all's")

de_stopwords = ["a","ab","aber","ach","acht","achte","achten","achter","achtes","ag","alle","allein","allem","allen","aller","allerdings","alles","allgemeinen","als","also","am","an","andere","anderen","andern","anders","au","auch","auf","aus","ausser","außer","ausserdem","außerdem","b","bald","bei","beide","beiden","beim","beispiel","bekannt","bereits","besonders","besser","besten","bin","bis","bisher","bist","c","d","da","dabei","dadurch","dafür","dagegen","daher","dahin","dahinter","damals","damit","danach","daneben","dank","dann","daran","darauf","daraus","darf","darfst","darin","darüber","darum","darunter","das","dasein","daselbst","dass","daß","dasselbe","davon","davor","dazu","dazwischen","dein","deine","deinem","deiner","dem","dementsprechend","demgegenüber","demgemäss","demgemäß","demselben","demzufolge","den","denen","denn","denselben","der","deren","derjenige","derjenigen","dermassen","dermaßen","derselbe","derselben","des","deshalb","desselben","dessen","deswegen","d.h","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","dir","doch","dort","drei","drin","dritte","dritten","dritter","drittes","du","durch","durchaus","dürfen","dürft","durfte","durften","e","eben","ebenso","ehrlich","ei","ei,","eigen","eigene","eigenen","eigener","eigenes","ein","einander","eine","einem","einen","einer","eines","einige","einigen","einiger","einiges","einmal","eins","elf","en","ende","endlich","entweder","er","Ernst","erst","erste","ersten","erster","erstes","es","etwa","etwas","euch","f","früher","fünf","fünfte","fünften","fünfter","fünftes","für","g","gab","ganz","ganze","ganzen","ganzer","ganzes","gar","gedurft","gegen","gegenüber","gehabt","gehen","geht","gekannt","gekonnt","gemacht","gemocht","gemusst","genug","gerade","gern","gesagt","geschweige","gewesen","gewollt","geworden","gibt","ging","gleich","gott","gross","groß","grosse","große","grossen","großen","grosser","großer","grosses","großes","gut","gute","guter","gutes","h","habe","haben","habt","hast","hat","hatte","hätte","hatten","hätten","heisst","her","heute","hier","hin","hinter","hoch","i","ich","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","im","immer","in","indem","infolgedessen","ins","irgend","ist","j","ja","jahr","jahre","jahren","je","jede","jedem","jeden","jeder","jedermann","jedermanns","jedoch","jemand","jemandem","jemanden","jene","jenem","jenen","jener","jenes","jetzt","k","kam","kann","kannst","kaum","kein","keine","keinem","keinen","keiner","kleine","kleinen","kleiner","kleines","kommen","kommt","können","könnt","konnte","könnte","konnten","kurz","l","lang","lange","leicht","leide","lieber","los","m","machen","macht","machte","mag","magst","mahn","man","manche","manchem","manchen","mancher","manches","mann","mehr","mein","meine","meinem","meinen","meiner","meines","mensch","menschen","mich","mir","mit","mittel","mochte","möchte","mochten","mögen","möglich","mögt","morgen","muss","muß","müssen","musst","müsst","musste","mussten","n","na","nach","nachdem","nahm","natürlich","neben","nein","neue","neuen","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nie","niemand","niemandem","niemanden","noch","nun","nur","o","ob","oben","oder","offen","oft","ohne","Ordnung","p","q","r","recht","rechte","rechten","rechter","rechtes","richtig","rund","s","sa","sache","sagt","sagte","sah","satt","schlecht","Schluss","schon","sechs","sechste","sechsten","sechster","sechstes","sehr","sei","seid","seien","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","selbst","sich","sie","sieben","siebente","siebenten","siebenter","siebentes","sind","so","solang","solche","solchem","solchen","solcher","solches","soll","sollen","sollte","sollten","sondern","sonst","sowie","später","statt","t","tag","tage","tagen","tat","teil","tel","tritt","trotzdem","tun","u","über","überhaupt","übrigens","uhr","um","und","und?","uns","unser","unsere","unserer","unter","v","vergangenen","viel","viele","vielem","vielen","vielleicht","vier","vierte","vierten","vierter","viertes","vom","von","vor","w","wahr?","während","währenddem","währenddessen","wann","war","wäre","waren","wart","warum","was","wegen","weil","weit","weiter","weitere","weiteren","weiteres","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wer","werde","werden","werdet","wessen","wie","wieder","will","willst","wir","wird","wirklich","wirst","wo","wohl","wollen","wollt","wollte","wollten","worden","wurde","würde","wurden","würden","x","y","z","z.b","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zu","zuerst","zugleich","zum","zunächst","zur","zurück","zusammen","zwanzig","zwar","zwei","zweite","zweiten","zweiter","zweites","zwischen","zwölf","euer","eure","hattest","hattet","jedes","mußt","müßt","sollst","sollt","soweit","weshalb","wieso","woher","wohin"]

def tokenize_words(text):
    tokens = text.split()
    # remove digits
    tokens = [token for token in tokens if not re.match(r'\b\d+\b', token)]
    return tokens

def clean_text(text, language, tokenize):
    if tokenize:
        words = []
    else:
        words = ""
    if isinstance(text, str):

        # remove emojis
        text = demoji.replace(text, repl="")

        # remove @s
        text = re.sub("@[A-Za-z0-9_]+", "", text)

        # remove hashtags
        text = re.sub("#[A-Za-z0-9_]+", "", text)

        # remove links
        text = re.sub('http://\S+|https://\S+', '', text)

        # remove encoded characters
        text = re.sub('&amp;', '', text)

        # replace '’' with '''
        text = re.sub('[’]+', '\'', text)

        # remove special characters
        text = re.sub('[\\\\!~•›\.=£►,*)@#%(&$_?^:;"|’“„”…‘/\+{}\[\]]+', ' ', text) # //so

        # replace '-' or '—' or '–' between any kind of space, with space
        text = re.sub('\s[\-—–]+\s', ' ', text) # --- -> 

        # remove '-' or '—' or '–' after or before words
        text = re.sub(r'(?!\b[-—–]+\b)(\b[-—–]+|[-—–]+\b)', '', text) # where-- -> where | -since -> since | gestapo-like -> gestapo-like

        # replace ''' between any kind of space, with space
        text = re.sub('\s+\'+\s+', ' ', text) # ' ->

        # remove ''' after words
        text = re.sub(r'\b\'+\s+|\b\'+\s+$', ' ', text) # alert' -> alert 

        # remove ''' before words
        text = re.sub(r'^\'+\b|\s+\'+\b', ' ', text) # 'pandering -> pandering

        # Lemmatization and remove stopwords
        if language == "en":
            stop_words = en_stopwords
            # remove unnecessary blank spaces
            words = tokenize_words(text.lower())
            words = [WordNetLemmatizer().lemmatize(word) for word in words if word not in (stop_words)]

        elif language == "de":
            stop_words = de_stopwords
            # remove unnecessary blank spaces
            text = " ".join(text.split())
            nlp = spacy.load('de_core_news_sm')
            lemma_text = nlp(text)
            words = [word.lemma_.lower() for word in lemma_text if word.lemma_ not in (stop_words) and not re.match(r'\b\d+\b', word.lemma_)]
        
        text = " ".join(words)

        # remove ''' after words
        words = re.sub(r'\b\'s', '', text) # trump's -> trump 

        if tokenize:
            # words = " ".join(words)
            words = words.split()

    return words

## GENERATE FRAQUENCIES
def count_frequencies(text):
    return text

# Output CSV file
# date_ranges = [['2021-01-01', '2021-02-01'], ['2021-02-01', '2021-03-01']]

date_ranges = [['2021-01-01', '2021-01-02']]
# date_ranges = [['2020-01-04', '2020-01-05']]

merge_tweets_by_date=True
# merge_tweets_by_date=False

languages = ['en']
# languages = ['en', 'de']

# file_exists = [False]
# file_exists = [False, False]

cols_to_drop = ['tweet', 'lang']
# cols_to_drop = ['lang']


for date_range in date_ranges:

    start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")
        print(f"Date: {date_str}")
        df_data = read_date_file(date_str)
        filtered_dfs = filter_dataframe(df_data, languages)
        for lang_idx, df_data in enumerate(filtered_dfs):
            print(f"Processing file: {date_str} / language: {languages[lang_idx]}")
            t = time()

            # Clean text column
            # df_data['tokenized_tweets'] = df_data['tweet'].apply(tokenize_words)
            # df_data["tokenized_tweets_n"] = df_data["tokenized_tweets"].apply(lambda n: len(n))
            # df_data["clean_tweets_n"] = df_data["clean_tweets"].apply(lambda n: len(n))

            filtered_df_by_lang = df_data['lang'] == languages[lang_idx]
            df_data["clean_tweets"] = df_data[filtered_df_by_lang]["tweet"].apply(lambda tweet: clean_text(tweet, languages[lang_idx], tokenize=False))

            output_file = f"./output/{languages[lang_idx]}_{date_str}_output.csv"
            # output_file = f"./output/{languages[lang_idx]}_output.csv"

            if merge_tweets_by_date:

                df_final_data = df_data.groupby('created_at', as_index=False)['clean_tweets'].apply(' '.join)

                # remove duplicates, leave only unique words
                # df_final_data["clean_tweets"] = df_final_data["clean_tweets"].apply(lambda words: ' '.join(set(words.split())))

                # Write processed data to output file
                # df_final_data.to_csv(output_file, mode='a', index=False, header=not file_exists[lang_idx])
                df_final_data.to_csv(output_file, mode='a', index=False, header=True)

            else:

                # Drop columns
                df_data = df_data.drop(cols_to_drop, axis=1)

                # Count Frequencies
                # df_data['freq_count'] = count_frequencies(df_data['clean_tweets'])

                # Write processed data to output file
                # df_data.to_csv(output_file, mode='a', index=False, header=not file_exists[lang_idx])
                df_data.to_csv(output_file, mode='a', index=False, header=True)

            print(f"saved file {output_file}")
            print('Time to build file: {} mins'.format(round((time() - t) / 60, 2)))

            # Set file_exists to True after writing the header once
            # file_exists[lang_idx] = True