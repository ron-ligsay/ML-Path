text = 'Hi this is a small sentence'

# we choose a sequence lenght
seq_len = 3

# split text into a list of words
words = text.split()

# make lines
lines = []
for i in range(seq_len, len(words) + 1):
    line = ' '.join(words[i-seq_len:i])
    lines.append(line)


# import Tokenizer from keras preprocessing text
from tensorflow.keras.preprocessing.text import Tokenizer
# instantiate tokenizer
tokenizer = Tokenizer()
# fit it on t he previous lines
tokenizer.fit_on_texts(lines)
# turn the lines into numeric sequence
sequences = tokenizer.texts_to_sequencces(lines)

print(tokenizer.index_word)
# or 
print(tokenizer.word_index)


# Import Dense, LSTM and Embedding layers
from tensorflow.keras.layers import Dense, LSTM, Embedding
model = Sequential()
# vocabulary size
vocab_size = len(tokenizer.index_word) + 1
# starting with an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=2)) 
# input_lenght = 2; means two words are passing through the embedding layer
# adding a LSTM layer
model.add(LSTM(8))
# adding a dense hidden layer
model.add(Dense(8, activation='relu'))
# adding an output layer with softmax
model.add(Dense(vocab_size, activation='softmax'))

# Text Prediction with LSTMs
# You're working with this small chunk of The Lord of The Ring quotes stored in the text variable:
# ================================================================================================
#  It is not the strength of the body but the strength of the spirit. 
#  It is useless to meet revenge with revenge it will heal nothing. 
#  Even the smallest person can change the course of history.
#  All we have to decide is what to do with the time that is given us. 
#  The burned hand teaches best. After that, advice about fire goes to the heart.

# Split text into an array of words 
words = text.split()

# Make sentences of 4 words each, moving one word at a time
sentences = []
for i in range(4, len(words)):
  sentences.append(' '.join(words[i-4:i]))

# Instantiate a Tokenizer, then fit it on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5],sequences[:5]))

# Import the Embedding, LSTM and Dense layer
from tensorflow.keras.layers import Embedding,LSTM, Dense

model = Sequential()

# Add an Embedding layer with the right parameters
model.add(Embedding(input_dim = vocab_size, input_length = 3, output_dim = 8, ))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='softmax'))
model.summary()

# That's a nice looking model you've built! 
# You'll see that this model is powerful enough to learn text relationships, 
# we aren't using a lot of text in this tiny example and our sequences are quite short. 
# This model is to be trained as usual, you would just need to compile it 
# with an optimizer like adam and use crossentropy loss. This is because we have modeled 
# this next word prediction task as a classification problem with all the unique words 
# in our vocabulary as candidate classes.

def predict_text(test_text, model = model):
  if len(test_text.split()) != 3:
    print('Text input should be 3 words!')
    return False
  
  # Turn the test_text into a sequence of numbers
  test_seq = tokenizer.texts_to_sequences([test_text])
  test_seq = np.array(test_seq)
  
  # Use the model passed as a parameter to predict the next word
  pred = model.predict(test_seq).argmax(axis = 1)[0]
  
  # Return the word that maps to the prediction
  return tokenizer.index_word[pred]