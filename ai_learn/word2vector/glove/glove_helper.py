import numpy as np

from config import *

glove_dir = nlp_tc_config.GLOVE_PATH
oov_number = 500
embedding_matrix = np.random.uniform(-1.0, 1.0, size=(500, 50)).astype(np.float32)

# embedding_matrix = np.random.rand(500, 50)


class GloveHelper:
    def __init__(self):
        self.wordVectors = np.load(glove_dir+'/wordVectors.npy')
        print('Loaded the word vectors!')

        self.wordsList = np.load(glove_dir+'/wordsList.npy')
        self.wordsList = [word.decode('UTF-8') for word in self.wordsList]
    def getVector(self, sentence):
        print("This is getVector method")

        vectors = []
        for word in sentence.split():
            try:
                word_index = self.wordsList.index(word.strip().lower())
                word_vector = self.wordVectors[word_index]
                vectors.append(word_vector)
            except Exception:
                print("Word: [", word, "] not in wvmodel! skip it directly")
                vectors.append(self.get_oov_vector(word.strip().lower()))
                print("oov vector ", self.get_oov_vector(word.strip().lower()))

        print("Finished!")
        return vectors

    def oov_index(self, word):
        # Replace this simple hash function with a more sophisticated approach if needed
        return hash(word) % oov_number  # Using Python's built-in hash function
    def get_oov_vector(self, word):
        index = self.oov_index(word)
        return embedding_matrix[index]

    def getIndex(self, sentence):
        print("This is test2 method")


if __name__ == '__main__':
    glove_helper =  GloveHelper()

    vectors = glove_helper.getVector("I like cats , 1add, alert32")
    print(vectors)