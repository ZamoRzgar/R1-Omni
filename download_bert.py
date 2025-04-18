from transformers import BertTokenizer, BertModel

# Set the target directory
local_bert_path = "C:/Users/zamor/models/bert-base-uncased"

# Download and save the tokenizer
print("Downloading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(local_bert_path)

# Download and save the model
print("Downloading BERT model...")
model = BertModel.from_pretrained("bert-base-uncased")
model.save_pretrained(local_bert_path)

print("BERT model and tokenizer downloaded successfully to:", local_bert_path)
