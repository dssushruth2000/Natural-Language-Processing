import tokenizers
from transformers import TFAutoModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from transformers import TFAutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
from tensorflow.keras.utils import to_categorical
import numpy as np
from math import sqrt


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


train_dataset = load_dataset("dair-ai/emotion", split="train")
tr_dataset = train_dataset.shuffle(seed=0)
tr_dataset = train_dataset[:1200]
print(tr_dataset)

tokenized_train = tokenizer(tr_dataset["text"] , max_length=512, truncation=True, padding="max_length", return_tensors="tf")


train_y = to_categorical(tr_dataset["label"])


print(len(tokenized_train["input_ids"]), len(train_y))

bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False

maxlen = 512  	
token_ids = Input(shape=(maxlen,), dtype=tf.int32, name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32, name="attention_masks")

bert_output = bert_model(token_ids,attention_mask=attention_masks)
dense_layer = Dense(64,activation="relu")(bert_output[0][:,0])
output = Dense(6,activation="softmax")(dense_layer)
model = Model(inputs=[token_ids,attention_masks],outputs=output)

model_summary = model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],train_y, batch_size=20, epochs=5)

te_dataset = load_dataset("dair-ai/emotion", split="test")
test_dataset = te_dataset.shuffle(seed=0)
test_dataset = te_dataset[:1200]
# print(test_dataset)

tokenized_test = tokenizer(test_dataset["text"], max_length=512, truncation=True, padding="max_length", return_tensors="tf")
test_y = to_categorical(test_dataset["label"])


test_score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]], test_y, verbose=0)
print("Test loss:", test_score[0])
print("Test accuracy:", test_score[1])

print(test_y.shape, train_y.shape)

predictions = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]]) 
predictions_shape = predictions.shape
print(predictions_shape)

# print(predictions)
# a = predictions[0]
# print(a)
# b= predictions[1]
# print(b)


predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_y, axis=1)


correct_predictions = np.where(predicted_labels == true_labels)[0]
incorrect_predictions = np.where(predicted_labels != true_labels)[0]



correct_examples = [(test_dataset["text"][i], true_labels[i], predicted_labels[i]) for i in correct_predictions[:10]]
incorrect_examples = [(test_dataset["text"][i], true_labels[i], predicted_labels[i]) for i in incorrect_predictions[:10]]



print("Correctly predicted examples:")
for example in correct_examples:
    print(example)

print("Incorrectly predicted examples:")
for example in incorrect_examples:
    print(example)



def cosine_similarity(a, b):
    return np.dot(a,b)/(sqrt(np.dot(a,a))*sqrt(np.dot(b,b)) )

sentence_pairs = [
    ("The tree shed its leaves.", "The flower bloomed brightly."),
    ("I like pizza.", "She enjoys pasta."),
    ("The sun is in orange color.", "The moon is in white color."),
    ("He read a book.", "She write a letter."),
    ("She cooks dinner.", "He washes the dishes."),
    ("The train whistled loudly.", "The bird sang sweetly."),
]

for context_1, context_2 in sentence_pairs:
    t = tokenizer([context_1, context_2], max_length=9, truncation=True, padding=True, return_tensors="tf")
    output = bert_model(t["input_ids"], attention_mask=t["attention_mask"])
    similarity = cosine_similarity(output[0][0][2], output[0][1][2])
    print(f"cosine_similarity between the following sentences:\n'{context_1}'\t'{context_2}': {similarity}\n")
