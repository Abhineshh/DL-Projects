import tensorflow as tf
import numpy as np
import os

# === 1. Download and load dataset ===
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f"Dataset length: {len(text)} characters")

# === 2. Text preprocessing ===
vocab = sorted(set(text))
vocab_size = len(vocab)
print(f"{vocab_size} unique characters")

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# === 3. Create training examples ===
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)

# === 4. Prepare batches ===
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# === 5. Build the model ===
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size, None), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs, outputs)

# === 6. Compile the model ===
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=BATCH_SIZE)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# === 7. Set up checkpoints ===
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# === 8. Train the model ===
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# === 9. Rebuild the model for generation (batch_size = 1) ===
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Restore latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
model.build(tf.TensorShape([1, None]))

# === 10. Text generation function ===
def generate_text(model, start_string):
    num_generate = 1000
    temperature = 1.0

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, axis=0)
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

# === 11. Generate and print output ===
print("\n=== Generated Text ===")
print(generate_text(model, start_string="ROMEO: "))
