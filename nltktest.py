from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk 
nltk.download('stopwords') 
nltk.download('punkt')

example_sentence = "this is an example showing off stop word filtration." 
stop_words = set(stopwords.words("english")) # stop words are unnecessary for meaning

# print(stop_words)
words = word_tokenize(example_sentence) #splits all the words 
filtered_sentence = [w for w in words if not w in stop_words]

# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)
# print(filtered_sentence)

ps = PorterStemmer()
example_words = ["python", "pythoner", "pythoning", 'pythoned', 'pythonly']
# for w in example_words:
#     print(ps.stem(w))
for w in words:
    print(ps.stem(w))
