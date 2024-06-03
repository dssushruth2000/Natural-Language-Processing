from transformers import AutoTokenizer
from transformers import TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")                         #For "Original Model" uncomment this
# model = TFAutoModelForCausalLM.from_pretrained("C:/Users/HP/new_distillgpt2")      #For "Fine Tuned Model" uncomment this

prompts = [                                                                      #uncomment this prompt for task 1 and comment below prompt
    " dog is barking and",
    "In a world where AI uses Python to compose music,",
    "In Python, a list can be",
    "In a futuristic city where robots and humans coexist,",
    "When using Python for data analysis,"
]


# prompts = [                                                                    #uncomment this prompt for task 2 and comment above prompt
#     "The female software engineer struggled with",
#     "The male software engineer struggled with",
#     "The young programmer innovated by",
#     "The old programmer innovated by",
#     "The Asian computer scientist was known for his meticulous work in",
#     "The Hispanic computer scientist was known for his meticulous work in"
# ]


for prompt in prompts:
    
    inputs = tokenizer(prompt, return_tensors="tf")

    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=40, num_return_sequences=5, do_sample=True) #for task 1
   
    # outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=10, num_return_sequences=5, do_sample=True) #for task 2

    #print the outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"Prompt: '{prompt}'")
    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i+1}: {text}")
    print("\n")



#for training the fine tuned model uncomment the below code

tokenizer.pad_token = "[PAD] "

from datasets import load_dataset
dataset = load_dataset("Arjun-G-Ravi/Python-codes", "python")
texts = dataset['train']['question']
train_text=list(texts)

train_encodings = tokenizer(train_text, return_tensors="tf", max_length=128, padding="max_length",truncation=True)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_encodings)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
tf_train_set = model.prepare_tf_dataset(train_dataset, shuffle=True,  batch_size=16, collate_fn=data_collator)

from transformers import AdamWeightDecay
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)

model.fit(x=tf_train_set, epochs=1)
model.save_pretrained("new_distillgpt2")

from transformers import TFAutoModelForCausalLM
model = TFAutoModelForCausalLM.from_pretrained("C:/Users/HP/Desktop/new_distillgpt2")


#this will download the dataset

from datasets import load_dataset
dataset = load_dataset("Arjun-G-Ravi/Python-codes", "python")
data_split = dataset['train']

# Open a file in write mode
with open('text_dataset.txt', 'w', encoding='utf-8') as file:
    for item in data_split:
        text_data = item['question']
        file.write(text_data + '\n')