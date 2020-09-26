import collections
import numpy as np
import pickle
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


data_words_articles = []
len_topic = 0
len_words = 0
len_articles = 0

words = []


# prablity matrices 

pr_pr_Z_pr_D_W = []  # P(z | d,  w)
pr_W_Z = []  # P(W | Z)
pr_Z_D = []  # P(z | d)
pr_P_D = []  # P( d )
pr_D_W = []  # P( d | w)
pr_W_D = []  # P(w | d)
pr_Z_D_W = []


def int_par(data_words_articles):
    
    # data set
    data_words_articles = data_words_articles[:, :]

    # number of topics as per in instruction
    len_topic = 10

    # words occurance within a roq
    
    len_words = len(data_words_articles[0])

    # articles - 3113
    
    len_articles = len(data_words_articles)

    # generating matrics with ones
    
    pr_Z_D = np.ones([len_topic, len_articles], dtype=np.float)  # P(z | d)
    pr_W_Z = np.ones([len_words, len_topic], dtype=np.float)  # P(z | d)
    pr_P_D = np.ones((len_articles,), dtype=np.float)
    pr_Z_D_W = np.ones([len_topic, len_articles, len_words], dtype=np.float)  # P(z | d)

    pr_W_D = np.ones([len_words, len_articles], dtype=np.float)  # P(z | d)
    pr_D_W = np.ones([len_words, len_articles], dtype=np.float)  # P(z | d)

    # Assigning random values 
    
    pr_Z_D = np.random.random(size=(len_topic, len_articles))
    pr_W_Z = np.random.random(size=(len_words, len_topic))

    # pr of documents 
    pr_P_D = 1.0 / len_articles
    for i in range(len_articles):
        pr_P_D[i] = pr_P_D

def common_word_list():
    
    l_words = []

	# merge list and words from the article
    
	for words_l in words_la:
		l_words = words_l + l_words

	# deleting less than 3 word element 
        
	l_words = remove_elements(list_of_words, 3)


	# most common words from list(i.e. take like 20%)
	counter = collections.Counter(l_words)
	words = [word for word, word_count in counter.most_common(len(l_words))]
	words = words[round(len(words) * 0.2):]

	length_words = len(words)

def matrix():

	# matrix where columns are words
    
	df_words = np.zeros([.len_articles, .len_words])

	# iterrate over words and documents subsequently
    
	for index_d in range(len_articles):
		for index_w in range(lenwords):
			if words[index_d, index_w] in words:
				df_words.loc[index_d, index_w] = df_words.loc[index_d, index_w] + 1


    data_words_articles = df_words
	print(df_words)


def data_loading(path):
    
    # pickle data set in form of tuple
    
    pickle_in = open(path, "rb")
    data_set = pickle.load(pickle_in)

    # articles (documents)
    
    documents = []

    for data in data_set:
        documents.append(data[4])

    print("Total number of documents: ", len(documents))
    pickle_in.close()

    return documents

def process_words():
    pickle_in = open('file.pickle', "rb")
    data_set = pickle.load(pickle_in)

    documents = []

    for data in data_set:
        documents.append(data[4])

    print("Total number of documents: ", len(documents))
    pickle_in.close()

    articles = load_data(file_path)
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))

    for article in articles:
        raw = article.lower()
        tokens = tokenizer.tokenize(raw)

        stopped_tokens = [i for i in tokens if not i in en_stop]

        words.append(stopped_tokens)


def delete_element(lst, k):
    counted = Counter(lst)
    return [el for el in lst if counted[el] >= k]


def E_step():
    """
    calculates E step
    """
    for d in range(len_articles):
        for w in range(len_words):
            norm = 0.0
            for z in range(len_topic):
                pr_Z_D_W[z, d, w] = pr_W_Z[w, z] * pr_Z_D[z, d]
                norm = norm + pr_Z_D_W[z, d, w]
                
            # normalizing 
                
            for z in range(length_topic):
                pr_Z_D_W[z, d, w] = pr_Z_D_W[z, d, w] / norm if norm != 0 else 0


def M_step():
    """
    calculate M - step
    """
    # update pr_W_Z
    pr_W_Z = np.zeros([len_words, len_topic], dtype=np.float)
    for z in range(len_topic):
        norm = 0.0
        for w in range(len_words):
            sum = 0.0
            for d in range(len_articles):
                sum = sum + data_words_articles[d, w] * pr_Z_D_W[z, d, w]
            pr_W_Z[w, z] = sum
            norm = norm + sum
        for w in range(len_words):
            pr_W_Z[w, z] = pr_W_Z[w, z] / norm if norm != 0 else 0

    # update P(z | d)
    for d in range(len_articles):
        for z in range(len_topic):
            s = 0
            for w in range(len_words):
                count = data_words_articles[d][w]
                s = s + count * pr_Z_D_W[z, d, w]
            pr_Z_D[z][d] = s
            pr_Z_D[z][d] = pr_Z_D[z][d] / np.sum(data_words_articles[d]) if np.sum(data_words_articles[d]) != 0 else 0


def prablity_D_W():
    """
    generate prabilities
    """
    # prability P(w,d)
    
    for d in range(lenarticles):
        norm = 0.0
        for w in range(len_words):
            sum = 0.0
            for z in range(len_topic):
                sum = sum + pr_W_Z[w, z] * pr_Z_D[z, d]
            pr_W_D[w, d] = sum
            norm = norm + sum
        for w in range(len_words):
            pr_W_D[w, d] = pr_W_D[w, d] / norm if norm != 0 else 0

    # prability P(d, w)
    for w in range(len_words):
        norm = 0.0
        for d in range(len_articles):
            pr_D_W[d, w] = pr_P_D[d] * pr_W_D[w, d]
            norm = norm + pr_D_W[d, w]

        for d in range(len_articles):
            pr_D_W[d, w] = pr_D_W[d, w] / norm if norm != 0 else 0


def log_likelihood():
    
    """
    calculate logs-likelihood
    """
    L = 0.0
    for d in range(len_articles):
        for w in range(len_words):
            for z in range(len_topic):
                L = L + pr_D_W[d, w] * (
                    np.log(pr_Z_D[z, d] * pr_W_Z[w, z]) if pr_Z_D[z, d] * pr_W_Z[w, z] != 0 else 0)
    print(L)


def aspects():
    """
    generate aspects from each document and print them 
    """
    list_of_topics = []
    for d in range(d):
        ind = np.argpartition(pr_Z_D[:, d], -10)[-10:]
        for i in ind:
            print(ind[i])


initialize_parameters()

# STARTS HERE
# calculation
for i in range(1000):
    generate_E_step()
    generate_M_step()
    calculate_pr_D_W()
    generate_log()

print_aspects()

