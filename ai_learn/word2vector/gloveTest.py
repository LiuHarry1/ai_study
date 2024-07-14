import numpy as np

wordsList = np.load('glove/wordsList.npy')
print('Loaded the word list!')

wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordVectors = np.load('glove/wordVectors.npy')
print('Loaded the word vectors!')
print(len(wordsList))
# print(wordsList)
print(wordVectors.shape)
wordsList = [word.decode('UTF-8') for word in wordsList]
baseballIndex = wordsList.index('OOV')
print(baseballIndex)
print(wordVectors[baseballIndex])

print('-', wordVectors[0])


