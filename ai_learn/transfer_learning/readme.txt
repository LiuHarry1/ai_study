https://harryliu.blog.csdn.net/article/details/107723526
https://harryliu.blog.csdn.net/article/details/134465987

https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder


Key Steps to Implement:
Tokenization: Breaking down sentences into words or subwords.
Word Embeddings: Converting each token (word) into a fixed-size vector representation.
Sentence Encoding: Combining the word embeddings into a single vector representing the sentence.

Yes, the Universal Sentence Encoder (USE) model, available on TensorFlow Hub, is designed to encode natural language sentences into a fixed-length vector representation. The reason it can take a string input is because the model includes a preprocessing step that converts the input string into a numerical form, typically referred to as tokenization and embedding.

Here's why and how it works:

Tokenization: When you input a string, the model first splits the text into tokens (e.g., words, subwords, or characters) depending on the tokenization strategy used. This is necessary because models cannot directly operate on raw text.

Embedding: After tokenization, the words or tokens are mapped to numerical embeddings. These embeddings are vectors of real numbers that represent words or subword units in a high-dimensional space. The Universal Sentence Encoder uses techniques like word embeddings or transformer-based embeddings to achieve this.

Encoding: Once the string is converted into embeddings, the encoder (which might use architectures like Transformer, LSTM, or CNNs) processes these embeddings to produce a fixed-length vector (sentence embedding). This vector can then be used in downstream tasks such as semantic similarity, classification, or clustering.

Why Models Take Numbers, Not Strings:
Neural networks can only process numbers, specifically floating-point tensors. So any text-based model, including the Universal Sentence Encoder, will internally convert strings to numeric vectors before performing the computations. This allows the model to leverage mathematical operations to learn relationships between words and their meanings.

In short, while the input to USE appears to be strings (for ease of use), the actual input to the neural network is always numerical after this preprocessing step.