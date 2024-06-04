from datasets import load_dataset
#to download the dataset i have done this separetly
train_dataset = load_dataset("dair-ai/emotion", split="train")
tr_dataset = train_dataset.shuffle(seed=0)
tr_dataset = tr_dataset.select(range(1200))
# Open a file in write mode
with open('dataset_train.txt', 'w', encoding='utf-8') as file:
    for item in tr_dataset:
        # Assuming each item in the dataset is a dictionary and you're interested in the 'text' field
        text = item["text"]
        label = item["label"]
        file.write(f"{text}\t{label}\n")


te_dataset = load_dataset("dair-ai/emotion", split="test")
test_dataset = te_dataset.shuffle(seed=0)
test_dataset = te_dataset.select(range(1200))
with open("dataset_test.txt", "w", encoding="utf-8") as file:
    for item in test_dataset:
        # Write each item to the file in the desired format
        text = item["text"]
        label = item["label"]
        file.write(f"{text}\t{label}\n")