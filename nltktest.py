from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk 
nltk.download('nps_chat') # chat corpus contains 10k posts from IM sessions
# nltk.download('stopwords') 
# nltk.download('punkt')

# example_sentence = "this is an example showing off stop word filtration." 
# stop_words = set(stopwords.words("english")) # stop words are unnecessary for meaning

# # print(stop_words)
# words = word_tokenize(example_sentence) #splits all the words 
# filtered_sentence = [w for w in words if not w in stop_words]

# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)
# print(filtered_sentence)

# ps = PorterStemmer()
# example_words = ["python", "pythoner", "pythoning", 'pythoned', 'pythonly']
# for w in example_words:
#     print(ps.stem(w))
# for w in words:
#     print(ps.stem(w))

posts = nltk.corpus.nps_chat.xml_posts()[:10000] #xml annotations for first 10000 IM posts


def dialogue_act_features(post): #define a method
    features = {}
    for word in nltk.word_tokenize(post):
        # print(word)
        features['contains({})'.format(word.lower())] = True
    return features
# print(posts[0].text)
# print(posts[0].get('class'))
#the 'class' feature gives 1 of 15 dialogue act types: emotion, statement, YNQuestion
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
# for post in posts:
#     print(post.text)
#     print(post.get('class'))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size] #train classifier on the first tenth of data
classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(test_set[0])
# print(nltk.classify.accuracy(classifier, test_set)) #calculates accuracy of classifier model on given test set

mytestset = [({'contains(who)': True, 'contains(are)': True, 'contains(you)': True,
               'contains(to)': True, 'contains(be)': True, 'contains(this)': True, 'contains(bad)': True}, 'Question')]
# print(nltk.classify.accuracy(classifier, mytestset))
print(classifier.classify({'contains(who)': True, 'contains(are)': True, 'contains(you)': True,
                     'contains(to)': True, 'contains(be)': True, 'contains(this)': True, 'contains(bad)': True}))
