from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk 
nltk.download('stopwords') 
nltk.download('punkt')

example_sentence = "this is an example showing off stop word filtration." 
stop_words = set(stopwords.words("english"))

# print(stop_words)
filtered_sentence = []
words = word_tokenize(example_sentence)
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)