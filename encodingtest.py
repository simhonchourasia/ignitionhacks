import tensorflow as tf
import os
print('bye')

embed = tf.saved_model.load('tmp/tfhub_modules/use_module')

print('hi')
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])
print(embeddings)
