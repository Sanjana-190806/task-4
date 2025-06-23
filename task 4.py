
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Example corpus
texts = [
    "hello how are you",
    "hello how is your day",
    "how are you doing today"
]

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

# Create sequences: [w1]→w2, [w1 w2]→w3, ...
input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

# Pad sequences & split inputs/labels
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.eye(total_words)[y]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=total_words, output_dim=64, input_length=max_seq_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=200, verbose=2)
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(seed_text, model, tokenizer, max_seq_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    next_index = np.argmax(predicted)
    return tokenizer.index_word[next_index]

# Example usage:
print(predict_next_word("hello how", model, tokenizer, max_seq_len))
def generate_text(seed_text, n_words, model, tokenizer, max_seq_len):
    text = seed_text
    for _ in range(n_words):
        next_word = predict_next_word(text, model, tokenizer, max_seq_len)
        text += " " + next_word
    return text

# Example:
print(generate_text("hello how", 5, model, tokenizer, max_seq_len))

