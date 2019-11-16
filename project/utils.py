import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': '/home/chatbot/assets/nlp_assets/intent_recognizer.pkl',
    'TAG_CLASSIFIER': '/home/chatbot/assets/nlp_assets/tag_classifier.pkl',
    'TFIDF_VECTORIZER': '/home/chatbot/assets/nlp_assets/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': '/home/chatbot/assets/nlp_assets/thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': '/home/chatbot/assets/nlp_assets/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    embeddings=dict()
    for line in open(embeddings_path):
  
      
      embeddings[line.split()[0]]=np.array(line.split()[1:]).astype("float32")

    return embeddings, list(embeddings.values())[0].shape[0]


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################

    returnvector=np.zeros(dim)
    count=0
    
    if len(question)==0:
      return returnvector
    
    for word in question.split():
      if word in embeddings: 
        returnvector=returnvector+embeddings[word]
        count+=1
        
    if count>0:
      returnvector=returnvector/count
      
    return returnvector

   
def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
