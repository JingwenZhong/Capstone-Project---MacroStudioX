import re
import pandas as pd
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import spacy  # spaCy for lemmatization. Alternatives: gensim.utils.lemmatize()
import gensim  # text processing framework
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath  # this is a free alternative to LIWC
import datetime
from typing import Generator, Union

# load nlp models
# define your own stop words to remove
# define your own stop words to remove
from nltk.corpus import stopwords
stop_words = set(
    ['likely', 'bill', 'fify', 'begin', "when's", "c's", 'afterwards', 'got', 'wont', 'consequently', 'about', 'vol',
     'didn', 'mostly', 'thorough', 'vols', 'may', "they'll", 'instead', 'entirely', 'hers', 'namely', "here's",
     'specify', 'seemed', 'so', 'throug', 'by', 'we', 'possibly', 'should', 'available', 'nearly', "it'd", 'specifying',
     'ought', 'unto', 'course', 'please', 'm', 'yet', 'means', 'such', 'wants', 'little', 'quickly', 'gone', 'run',
     'outside', 'show', 'itself', 'arent', 'de', 'nos', 'everyone', 'un', 'mug', 'isn', 'their', 'nowhere', 'apparently',
     'proud', 'would', 'my', 'theyd', 'also', 'gets', 'theirs', 'pp', 'into', 'therere', 'com', 'obtained', 'regarding',
     'e', 'present', 'during', 'usefulness', "they're", 'fire', 'sec', 'moreover', 'que', 'him', 'makes', 'hello',
     'thanx', 'thin', "he'll", 'taking', 'does', 'anywhere', 'im', 'within', 'whole', 'following', "wasn't", 'whod',
     'under', 'if', 'these', "'ll", 'able', 'tip', 'hereby', 'youd', 'wed', 'its', 'appear', 'corresponding', 'who',
     'beginnings', 'normally', 'seeming', 'usually', 'couldnt', 'too', 'what', 'ok', 'keeps', 'thanks', 'hasnt', 'mill',
     'serious', "how's", 'others', "we've", 'wherever', 'down', 'via', 'mustn', 'anyways', 'hi', 'index', "'ve", 'tried',
     'youre', 'concerning', 'do', 'or', 'whom', 'd', "that'll", 'h', 'neither', 'least', "she'll", 'therein', 'reasonably',
     'never', 'twelve', 'ltd', 'usefully', 'saying', 'ah', 'could', 'owing', 'like', 'keep', 'of', 'resulted', 'whereafter',
     'empty', 'forty', 'date', 'g', "should've", 'otherwise', 'insofar', 'downwards', 'id', 'b', 'except', 'thoroughly',
     'enough', 'effect', 'potentially', 'zero', 'seven', 'come', 've', "i'll", 'twice', 'thats', 'non', 'important',
     "hadn't", 'thereof', 'four', 'ed', 'although', 'new', 'rd', 'follows', 'j', 'gotten', 'k', 'despite', "let's", 'thus',
     'along', 'ref', 'hed', 'mg', 'inward', "you'll", 'way', 'ml', 'approximately', 'everything', 'obviously', 'relatively',
     'n', 'whither', 'still', 'back', 'an', "i'm", 'nobody', 'those', "why's", 'kept', 'find', "don't", 'particular', 'a',
     'particularly', "aren't", "can't", 'unlike', 'liked', 'eleven', 'maybe', "who'll", "she's", 'very', 'resulting',
     "there'll", "mightn't", 'been', 'without', 'become', 'whim', 'did', 'sure', 'nothing', 'becomes', 'below', 'km',
     'provides', 'front', 'given', 'adj', 'whenever', 'giving', 'part', 'far', 'known', 'sent', 'x', 'hasn', 'appropriate',
     "you'd", 'alone', 'ending', 'miss', 'last', 'through', 'exactly', 'novel', 'asking', 'onto', 'p', 'am', 'plus', 'showns',
     'detail', 'until', 'noone', 'two', 'self', 'inc', 'whoever', 'somewhat', 'sufficiently', 'several', 'oh', 'indeed',
     'name', 'itd', 'can', 'having', 'to', 'y', 'somehow', 'hid', "you're", 'na', 'cant', 'strongly', 'information',
     'before', 'own', 'uses', 'unlikely', "shouldn't", 'better', 'soon', 'mr', 'forth', 'hence', 'six', 'getting',
     'describe', 'most', "we'd", 'importance', 'see', 'truly', 'himself', 'every', 'together', 'useful', 'doesn',
     'nonetheless', 'specified', 'omitted', 'nine', 'containing', "we'll", 'almost', 'at', 'probably', 'towards',
     'across', 'fill', 'brief', 'howbeit', 'look', 'both', 'comes', 'be', 'respectively', 'couldn', 'similar', 'thank',
     'previously', 'largely', "hasn't", 'go', 'primarily', 'abst', 'more', 'arise', 'thou', 'mightn', 'thered', 'wish',
     'taken', 'myself', 'try', 'ex', 'many', 'act', 'merely', 'right', 'mine', "it'll", 'value', 'haven', 'greetings',
     "t's", 'related', 'your', 'sometimes', 'cry', 'another', 'done', 'behind', 'besides', 'while', 'us', 'everybody',
     'somewhere', 'wasnt', 'secondly', 'no', 'perhaps', 'willing', "shan't", 'five', 'aren', 'took', 'use', "ain't",
     'wasn', 'sup', 'even', 'fifth', 'gave', 'that', 'results', 'v', 'obtain', 'unless', 'give', 'line', 'whereas',
     'hereupon', "c'mon", 'meantime', 'anymore', 'sensible', 'eighty', 'l', 's', 'made', 'widely', 'www', 'indicates',
     'later', 'is', 'put', 'being', 'her', 'especially', 'looking', 'some', 'etc', 'indicated', "he'd", 'had', 'sometime',
     'allow', 'inasmuch', 'interest', 'top', 'using', 'shes', 'added', 'showed', 'know', 'on', 'needs', 'possible',
     'certain', 'page', 'help', 'predominantly', 'call', 'seems', 'wouldnt', 'few', 'mean', 'beforehand', 'somethan',
     'because', 'thousand', 'apart', 'beside', 'well', 'followed', 'anyhow', 'according', 'lets', 'c', 'formerly',
     'immediately', 'slightly', 'anyway', 'similarly', 'vs', 'in', "you've", 'quite', 'beginning', 'might', 'must',
     'allows', 'suggest', "needn't", 'old', "that've", 'weren', 'happens', 'full', 'thereto', 'hopefully', 'when', 'ts',
     'z', 'per', 'significant', 'were', 'thereafter', 'research', "they'd", 'and', 'first', 'wouldn', 'presumably',
     'wonder', 'wheres', 'twenty', 'knows', 'home', 'hadn', 'bottom', 'something', 'past', "weren't", 'everywhere',
     'ten', "doesn't", 'readily', 'seen', 'nay', "he's", 'w', 'whereby', 'latterly', 'anything', 'til', 'above',
     'hereafter', 'tell', 'various', 'becoming', 'though', 'selves', 'someone', 'either', 'hither', 'any', 'actually',
     'whose', 'less', 'whence', 'shan', "there's", 'theres', 'ignored', 'between', 'unfortunately', 'however', 'million',
     'shouldn', 'don', 'example', 'much', "it's", 'affects', 'pages', "a's", 'move', 'thereupon', 'viz', 'ca', 'lately',
     'meanwhile', 'once', 'ord', 'looks', 'herein', 'r', "haven't", 'against', 'from', "what'll", 'need', 'biol', 'thoughh',
     'let', "i'd", 'aside', 'refs', "didn't", 'yourselves', 'ninety', 'co', "mustn't", 'say', 'hardly', 'them', "there've",
     'it', "she'd", 'anybody', 'somebody', 'considering', 'now', 'take', 'furthermore', 'after', 'briefly', 'latter', 'th',
     'whos', 'con', 'fix', 'herself', 'announce', 'get', 'ff', 'heres', 'regardless', 'thru', 'placed', 'changes', 'side',
     'sincere', 'sixty', "won't", 'ma', 'kg', 'again', 'our', 'et', 'whether', 'described', 'nor', 'with', 'this', 'since',
     'words', 'shows', 'best', 'saw', 'have', 'went', 'going', 'anyone', 'same', 'which', 'said', 'welcome', 'necessarily',
     'sub', "couldn't", 'throughout', 'mrs', 'ran', 'rather', 'only', 'the', 'contains', 'she', 'than', 'yes', 'different',
     'begins', 'me', 'around', 'ones', 'theyre', 'appreciate', 'but', 'immediate', 'seriously', 'he', 'thickv', 'just',
     'stop', 'why', 'always', 'was', "wouldn't", 'auth', 'next', 'has', 'sorry', 'mainly', 'over', "where's", 'edu', 'qv',
     'up', 'affected', 'accordance', 'ain', 'as', 'affecting', 'associated', 'you', 'found', 'shed', 'whereupon', 'second',
     'overall', 'three', 'cannot', 'where', 'recent', 'yourself', 'away', 'f', 'wherein', 'system', 'regards', 'other',
     'became', 'significantly', 'trying', 'then', 'fifteen', 'ourselves', 'clearly', 'whatever', 'further', 'not', 'believe',
     'causes', 'former', 'came', 'contain', 'end', 'll', 'for', 't', 'out', 'invention', 'nevertheless', "they've", 'recently',
     'seeing', "what's", 'promptly', 'among', 'needn', 'used', 'section', 'want', 'certainly', 'else', "that's", 'therefore',
     'nd', 'near', 'his', 'shown', 'o', 'noted', 'already', 'eight', 'each', 'necessary', 'upon', 'definitely', 'amongst',
     'eg', 'whats', 'ours', 'u', 'thereby', 'inner', 'consider', 'successfully', 'will', 'ask', 'seem', 'ups', 'themselves',
     "who's", 'won', 'are', 'ie', 'off', 'one', 'says', 'tends', 'hundred', 'yours', 'think', 'i', 'lest', 'how', 'all',
     'often', 'werent', 'amount', 'indicate', 'currently', 'goes', 'there', 'thence', 'toward', 'tries', 'world', 'cause',
     'make', 'accordingly', 'okay', 'elsewhere', 'none', 'shall', 'specifically', 'due', 'third', 'here', 'beyond', 're',
     "i've", 'amoungst', 'really', 'whomever', 'they', 'gives', 'q', 'hes', "we're", 'substantially', 'ever', "isn't", 'doing',
     'awfully', 'poorly ',"kinda","guys", "lol",'shakalaka','amp', 'fuck', 'looooooooool','haha','lmfao','lmao'
     'ouch', 'shit','bruh', '•', 'damn', 'thx','sir','wassup','afaik','asap','cya','omg','₺','म','ो','द','ी','ह',
     'ै','त','ो','बर','्','ब','ा','द','ी','ह','ै','‼','ass','®','wtf','thks','suck','dude'
     ])
stop_words.update(stopwords.words('english'))  # add all of nltk's stop words
stop_words.update(stopwords.words('english'))  # add all of nltk's stop words

abbreviations = {
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
     "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired",
    "%" : "percent"
}


# Target parts of speech
# complete list at: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
target_pos = set(['FW',  # Foreign word
                  'JJ',  # Adjective
                  'JJR',  # Adjective, comparative
                  'JJS',  # Adjective, superlative
                  'NN',  # Noun, singular or mass
                  'NNS',  # Noun, plural
                  'NNP',  # Proper noun, singular
                  'NNPS',  # Proper noun, plural
                  'RB',  # Adverb
                  'RBR',  # Adverb, comparative
                  'RBS',  # Adverb, superlative
                  'SYM',  # Symbol
                  'UH'  # Interjection
                  ])

nlp = spacy.load('en_core_web_sm', disable=['parser',
                                            'ner'])  # Initialize spacy English model, keeping only tagger component (for efficiency)
lexicon = Empath()  # initialize Empath object
sid_obj = SentimentIntensityAnalyzer()  # initialize vader object


def find_retweeted(tweet: str) -> list:
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)


def find_mentioned(tweet: str) -> list:
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)


def find_hashtags(tweet: str) -> list:
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)


def find_emojis(text: str, rm: bool = False) -> Union[list, str]:
    '''Takes a string and either finds or removes emoticons'''

    # specify the UNICODE values of emojis to be removed from tweet
    regrex_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\U0001F700-\U0001F77F"  # alchemical symbols
                                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                u"\U00002702-\U000027B0"  # Dingbats
                                u"\u2640-\u2642" 
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                u"\u2066"
                                "]+", re.UNICODE)

    # if input specifies the removal of emojis
    if rm:
        return regrex_pattern.sub(r'', text)
    else:
        return regrex_pattern.findall(text)


def find_urls(text: str, rm: bool = False) -> Union[list, str]:
    '''Takes a string and either finds or removes web addresses '''
    if rm:
        return re.sub(r'https?://\S+', '', text)
    else:
        return re.findall(r'https?://\S+', text)


def remove_query_terms(data: list, keywords: list) -> list:
    '''
    Takes a list of strings and removes keywords.
    keywords is a list of unprocessed search terms. e.g. ['#inflation','@federalreserve']
    '''
    # For each string in input data:
    # split() each tweet string on spaces
    # remove key words
    # ' '.join() to combine words back into sentences
    return [' '.join([word for word in tweet.split() if word not in keywords]) for tweet in data]


def remove_whitespace(text: str) -> str:
    '''Takes a string and returns that string with training whitespace and newline characters removed '''
    return re.sub(r'\s+', ' ', text)


def remove_apostrophe(text: str) -> str:
    '''Takes a string and returns that string with all apostrophes removed '''
    return re.sub(r'\'', '', text)


def remove_handles(text: str) -> str:
    '''Takes a string and returns that string with all twitter handles and mentions removed '''
    return re.sub(r'\.?@\S*', '', text)


def expand_contractions(text: str) -> str:
    '''Takes a string and returns that string with some select contractions expanded e.g. "can't" -> "can not" '''
    # specific contractions
    text = re.sub(r"\bwon\'t\b", "will not", text)
    text = re.sub(r"\bcan\'t\b", "can not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\bain\'t\b", "are not",
                  text)  # This ignores the alternatives: ["am not", "are not", "is not", "has not", "have not"]
    text = re.sub(r"\bdunno\b", "do not know", text)
    text = re.sub(r"\bsome1\b", "someone", text)
    text = re.sub(r"\bhahah|hahaha|hahahaha\b", "haha", text)
    text = re.sub(r"\blmao|lolz|rofl\b", "lol", text)
    text = re.sub(r"\bthanx|thnx\b", "thanks", text)
    text = re.sub(r"\bgoood\b", "good", text)
    text = re.sub(r"\b4got|4gotten\b", "forget", text)
    text = re.sub(r"\b2day\b", "today", text)
    text = re.sub(r"\b2morow|2moro\b", "tomorrow", text)
    text = re.sub(r"\byrs\b", "years", text)
    text = re.sub(r"\bhrs\b", "hours", text)

    # general contractions
    text = re.sub(r"\Bn\'t\b", " not",
                  text)  # \B looks for \w characters. \b looks for the edges of word characters (note: it sees the apostrophe as a \W character and thinks it's the end of the word)
    text = re.sub(r"\b\'re\b", " are", text)
    text = re.sub(r"\b\'s\b", " is", text)  # This has limitations with possessives and "has"
    text = re.sub(r"\b\'d\b", " would", text)  # This has limitations with "I had", "I could", etc.
    text = re.sub(r"\b\'ll\b", " will", text)
    text = re.sub(r"\b\'ve\b", " have", text)
    text = re.sub(r"\b\'m\b", " am", text)
    # Additional contractions:
    # https://github.com/ian-beaver/pycontractions/blob/6023c637956a5fa1e2cf8714aa53e10cfb807305/pycontractions/contractions.py/

    return text


def clean_strings(data: list, rm_emojis: bool = True) -> list:
    '''Takes a list of strings and removes some special characters'''
    # documentation for regular expressions: https://docs.python.org/3/howto/regex.html

    # Remove trailing spaces and newline characters from each tweet
    data = [remove_whitespace(tweet_str) for tweet_str in data]

    # Remove all web addresses (http... and https...) from each tweet
    data = [find_urls(tweet_str, rm=True) for tweet_str in data]

    # Remove emojis from each tweet
    if rm_emojis:
        data = [find_emojis(tweet_str, rm=True) for tweet_str in data]

        # Expand contractions
    data = [expand_contractions(tweet_str) for tweet_str in data]

    # Remove any remaining apostrophes
    data = [remove_apostrophe(tweet_str) for tweet_str in data]

    # Remove all twitter handles (e.g. @federalreserve)
    data = [remove_handles(tweet_str) for tweet_str in data]

    # ````````````````Optional```````````````````
    # Remove any @ addresses including emails and twitter handles
    # data = [re.sub(r'\S*@\S*\s?', '', tweet_str) for tweet_str in data]
    return data


def tokenization(data: list) -> Generator:
    '''
    Takes a list of strings and returns a generator object to create tokenized word-lists
    Note: Call list(tokenization(data)) to get final result from this function
    '''
    for tweet_str in data:
        # simple_preprocess() tokenizes tweet string.
        # Splits special characters from strings, converts to lowercease UNICODE, removes accents and punctuation(:, @, [comma], etc.), and sets word length limits.
        yield (gensim.utils.simple_preprocess(str(tweet_str),
                                              min_len=2,  # minimum word length
                                              max_len=20,  # maximum word length
                                              deacc=True))  # deacc=True removes accents
    # Alternatively, use nltk for lighter preprocessing during tokenization
    # e.g.:
    # from nltk.tokenize import TweetTokenizer
    # tknzr = TweetTokenizer()
    # tknzr.tokenize("The code didn't work!")
    # [u'The', u'code', u"didn't", u'work', u'!']
    # another example:
    # tknzr.tokenize('From March 5th. 👇https://t.co/hv136r4lV8 \n\nAgain, everything happening was totally predictable. \n\n#inflation')
    # ['From', 'March', '5th', '.', '👇', 'https://t.co/hv136r4lV8', 'Again', ',', 'everything', 'happening', 'was', 'totally', 'predictable', '.', '#inflation']


def remove_stopwords(tweet_list: list, stop_words: list) -> list:
    '''Takes a list of tokenized documents and returns the same structure without stop words'''
    return [[word for word in tweet if word not in stop_words] for tweet in tweet_list]


def remove_pos(tweet_list: list, target: list = target_pos) -> list:
    '''Takes a list of token lists as input and uses the nltk part of speach tagger to remove the targetted part of speach'''
    data_pos = [nltk.pos_tag(tweet) for tweet in
                tweet_list]  # nltk.pos_tag() returns a list of tuples with (word,pos) pairs.
    data_pos = [[token for token, pos in tweet if pos in target] for tweet in data_pos]
    return data_pos


def lemmatization(tweet_list: list, nlp, stop_words: list,
                  allowed_postags: list = ['NOUN', 'ADJ', 'VERB', 'ADV']) -> list:
    '''
    Input:
    tweet_list: list of tokenized documents
    nlp: spaCy english language model
    Output:
    Removes erroneous parts of speech and performs lemmatization on each token
    '''
    return [[token.lemma_ for token in nlp(" ".join(tweet)) if
             token.pos_ in allowed_postags and token.lemma_ not in stop_words] for tweet in tweet_list]



def pre_process(df: pd.DataFrame, keywords: str = "", rm_emojis: bool = False, filter_pos: bool = False,
                lemm: bool = False) -> list:
    '''
    Inputs:
    df: dataframe of tweets. Must have a "text" column
    keywords: list of unprocessed search terms. e.g. ['#inflation','@federalreserve']
    Output:
    returns a list of strings containing the cleaned text
    '''
    # "data" is a list of each tweet's text as a string
    data = df.text.values.tolist()

    # remove query terms from list of strings
    if keywords != "":
        data = remove_query_terms(data, keywords)

    # clean each tweet's text
    data = clean_strings(data, rm_emojis=rm_emojis)

    # Tokenize each sentence string
    data = list(tokenization(data))

    if filter_pos:
        # remove stop words
        data = remove_stopwords(data, stop_words)

        # Remove everything but nouns, adj, and adverbs.
        data = remove_pos(data)

    if lemm:
        # Remove everything but nouns, adj, and adverbs.
        print(f"{datetime.datetime.now()} - Starting lemmatization.")
        data = lemmatization(data, nlp, stop_words)
        print(f"{datetime.datetime.now()} - Finished lemmatization.")

    return [' '.join(doc) for doc in data]


###``````````````````````````````

def remove_retweets(df: pd.DataFrame) -> pd.DataFrame:
    '''Takes a dataframe of tweets and returns a new dataframe with no records of retweets'''
    return df[df.text.str[:3] != "RT "].copy()


def build_bigram_models(data_words: list, min_count=5, threshold=10):
    '''Takes an iterable of token lists and returns a generator object trained on frequent bigrams in the corpus'''
    bigram = gensim.models.Phrases(data_words, min_count=min_count,
                                   threshold=threshold)  # higher threshold, fewer phrases.
    bigram_model = gensim.models.phrases.Phraser(bigram)

    return bigram_model


def all_bigrams(a_list: list) -> list:
    ''' Takes a list of tokens as input and returns a list of all bigrams in the list '''
    return [f"{a}_{b}" for a, b in zip(a_list[:-1], a_list[1:])]


def all_trigrams(a_list: list) -> list:
    ''' takes a list of tokens as input and returns a list of all trigrams in the list '''
    return [f"{a}_{b}_{c}" for a, b, c in zip(a_list[:-2], a_list[1:-1], a_list[2:])]


def make_freq_bigrams(tweet_list: list, bigram_mod) -> list:
    '''
    Input:
        tweet_list: list of tokenized documents
        bigram_mod: pretrained bigram generator object
    Output:
        corpus of tokenized documents with bigrams included
    '''
    return [bigram_mod[tweet] for tweet in tweet_list]


def process_text(data: list, ngrams: str) -> list:
    '''
    Input:
        data: list of strings
    Output:
        perform text transformation and return list of processed strings
    '''
    # split text strings into tokens
    data = [doc.split() for doc in data]

    # form n-grams
    if ngrams == "freq":
        bigram_mod = build_bigram_models(data)  # using corpus with no retweets.
        data_words_ngrams = make_freq_bigrams(data, bigram_mod)
    elif ngrams == "bi":
        data_words_ngrams = list(map(all_bigrams, data))
    elif ngrams == "tri":
        data_words_ngrams = list(map(all_trigrams, data))
    else:
        data_words_ngrams = data

    return [' '.join(doc) for doc in data_words_ngrams]  # join tokens in each doc to make a list of strings


# ````````````````````````

def get_polarity(tweet: str, margin: float = 0.05) -> int:
    ''' Takes the tweet text (str) and VADER sentiment object and returns polarity = {1,0,-1}'''

    # analyze tweet text
    sentiment_dict = sid_obj.polarity_scores(tweet)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= margin:
        return 1
    elif sentiment_dict['compound'] <= - margin:
        return -1
    else:
        return 0


def get_subjectivity(tweet: str) -> int:
    ''' Takes the tweet text (str) and VADER sentiment object and returns subjectivity between 0 and 1'''
    # Alternatively, we could use TextBlob for subjectivity analysis

    # analyze tweet text
    sentiment_dict = sid_obj.polarity_scores(tweet)

    # decide subjectivity of the tweet
    if 1 - abs(sentiment_dict['compound']) >= 0.5:
        return 1
    else:
        return 0


def get_liwc_emotions(text: str, categories: list) -> dict:
    # apply lexicon to each tweet text

    if isinstance(text, str):  # if this tweet has any clean_text
        # get emotions counts
        res = lexicon.analyze(text, categories=categories, tokenizer="default", normalize=False)
        # total word count in this tweet:
        res['word_count'] = len(text.split())
    else:
        res = dict.fromkeys(categories, 0)  # dictionary of 0s using all the column names from previous tweet
        res['word_count'] = 0
    return res

# ``````````````````````
