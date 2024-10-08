from transformers import BertTokenizer, BertForNextSentencePrediction, GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load BERT tokenizer and model for NSP (using pre-trained directly)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# Load GPT-2 tokenizer and model for next sentence generation (using pre-trained directly)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
