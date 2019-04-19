from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk 
nltk.download('nps_chat') # chat corpus contains 10k posts from IM sessions

posts = nltk.corpus.nps_chat.xml_posts()[:10000] #xml annotations for first 10000 IM posts

def dialogue_act_features(post): #define a method
    features = {}
    for word in nltk.word_tokenize(post):
        # print(word)
        features['contains({})'.format(word.lower())] = True
    return features

# print(posts[0].text)
# print(posts[0].get('class'))
#the 'class' feature gives 1 of 15 dialogue act types: emotion, statement, 
#    YNQuestion, WHQuestion, Greet, Clarify, Other
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]

size = int(len(featuresets) * 0.9)
train_set, test_set = featuresets[size:], featuresets[:size] #train classifier on the first tenth of data
# classifier = nltk.NaiveBayesClassifier.train(featuresets)
# print(test_set[0])
# print(nltk.classify.accuracy(classifier, test_set)) #calculates accuracy of classifier model on given test set

# mytestset = [({'contains(did)': True, 'contains(you)': True, 'contains(eat)': True,
            #    'contains(yet)': True}, 'Question')]

txt = input("Type a sentence for me to classify: ")
mytestset = dialogue_act_features(txt)

# print(nltk.classify.accuracy(classifier, mytestset))
# classifier.show_most_informative_features()
# print(classifier.classify({'contains(did)': True, 'contains(you)': True, 'contains(eat)': True,
                        #    'contains(yet)': True}))

#maxent classifier
maxent_classifier = nltk.MaxentClassifier.train(train_set, max_iter=4)
# print(maxent_classifier.classify(test_set[0][0]))
print(maxent_classifier.classify(mytestset))
# maxent_classifier.show_most_informative_features()
# print(nltk.classify.accuracy(maxent_classifier, test_set))
