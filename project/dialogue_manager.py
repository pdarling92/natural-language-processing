import os
from sklearn.metrics.pairwise import pairwise_distances_argmin, cosine_similarity

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

from utils import *



class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        question_vec = qvec=question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        similarity=cosine_similarity([qvec],thread_embeddings )
        best_thread = np.argmax(similarity)

        return thread_ids[best_thread]

class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

#        self.chatbot=create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        self.chatbot = ChatBot('Chatbotty McBotFace')

        # Create a new trainer for the chatbot
        trainer = ChatterBotCorpusTrainer(self.chatbot)

        # Train the chatbot based on the english corpus
        trainer.train("chatterbot.corpus.english")  

        return(self.chatbot)

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)#### YOUR CODE HERE ####

        vectorizer = unpickle_file(RESOURCE_PATH['TFIDF_VECTORIZER'])
        intent_recognizer = unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])
        tag_classifier=unpickle_file(RESOURCE_PATH['TAG_CLASSIFIER'])

        features = vectorizer.transform([prepared_question]) #### YOUR CODE HERE ####
        intent = intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(question) #### YOUR CODE HERE ####
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = tag_classifier.predict(features)[0] #### YOUR CODE HERE ####
            
            # Pass prepared_question to thread_ranker to get predictions.

            if len(tag)> 0:
              thread_id=self.thread_ranker.get_best_thread(prepared_question,tag)
            else:
              thread_id = -1

            return self.ANSWER_TEMPLATE % (tag, thread_id)
