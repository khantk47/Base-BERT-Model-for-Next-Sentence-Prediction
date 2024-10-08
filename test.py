import json
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import psutil
import torch
from transformers import BertForNextSentencePrediction, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import time

def load_bert_model(model_name):
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_gpt2_model(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict_nsp(model, tokenizer, sentence_a, sentence_b):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    follows = probs[0] > probs[1]
    return follows, probs[0].item()

def generate_next_sentence(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    start_time = time.time()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    inference_time = time.time() - start_time
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    next_sentence = generated_text[len(input_text):].strip()
    
    return next_sentence, inference_time

def test_nsp_and_generate(bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer, sentence_pairs, expected_results):
    nsp_predictions = []
    generation_inference_times = []
    correct_predictions = 0
    total_predictions = len(sentence_pairs)
    gpt2_perplexities = []
    
    for (sentence_a, sentence_b), expected in zip(sentence_pairs, expected_results):
        # NSP prediction
        follows, prob = predict_nsp(bert_model, bert_tokenizer, sentence_a, sentence_b)
        nsp_predictions.append(int(follows))
        
        if int(follows) == expected:
            correct_predictions += 1
        
        # GPT-2 generation
        generated_next, gen_inf_time = generate_next_sentence(gpt2_model, gpt2_tokenizer, sentence_a)
        generation_inference_times.append(gen_inf_time)
        
        # Calculate perplexity for GPT-2
        input_ids = gpt2_tokenizer.encode(sentence_b, return_tensors='pt')
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        gpt2_perplexities.append(perplexity.item())
        
        print(f"Input A: {sentence_a}")
        print(f"Input B: {sentence_b}")
        print(f"NSP Prediction: {'Follows' if follows else 'Does not follow'} (Probability: {prob:.4f})")
        print(f"Expected: {'Follows' if expected else 'Does not follow'}")
        print(f"Generated next sentence: {generated_next}")
        print(f"Generation time: {gen_inf_time:.5f} seconds")
        print(f"Perplexity: {perplexity.item():.4f}")
        print()

    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    error_rate = 1 - accuracy
    precision = precision_score(expected_results, nsp_predictions)
    recall = recall_score(expected_results, nsp_predictions)
    f1 = f1_score(expected_results, nsp_predictions)
    auc_roc = roc_auc_score(expected_results, nsp_predictions)
    avg_inference_time = np.mean(generation_inference_times)
    throughput = 1 / avg_inference_time

    # Get resource utilization
    bert_model_size = sum(p.numel() for p in bert_model.parameters()) * 4 / (1024 * 1024)  # Size in MB
    gpt2_model_size = sum(p.numel() for p in gpt2_model.parameters()) * 4 / (1024 * 1024)  # Size in MB
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Memory usage in MB

    # GPT-2 specific metrics
    avg_gpt2_perplexity = np.mean(gpt2_perplexities)
    gpt2_throughput = 1 / np.mean(generation_inference_times)

    # Prepare output metrics
    output_metrics = {
        "performance_metrics": {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc
        },
        "inference_metrics": {
            "avg_inference_time": avg_inference_time,
            "throughput": throughput
        },
        "resource_utilization": {
            "bert_model_size": bert_model_size,
            "gpt2_model_size": gpt2_model_size,
            "memory_usage": memory_usage
        },
        "gpt2_metrics": {
            "perplexity": avg_gpt2_perplexity,
            "avg_inference_time": np.mean(generation_inference_times),
            "throughput": gpt2_throughput
        }
    }

    # Save output metrics to JSON file
    with open('output_metrics.json', 'w') as f:
        json.dump(output_metrics, f, indent=4)

    print("Summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Average inference time: {avg_inference_time:.5f} seconds")
    print(f"Throughput: {throughput:.2f} inferences/second")
    print(f"BERT Model Size: {bert_model_size:.2f} MB")
    print(f"GPT-2 Model Size: {gpt2_model_size:.2f} MB")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"GPT-2 Average Perplexity: {avg_gpt2_perplexity:.4f}")
    print(f"GPT-2 Average Inference Time: {np.mean(generation_inference_times):.5f} seconds")
    print(f"GPT-2 Throughput: {gpt2_throughput:.2f} inferences/second")

if __name__ == "__main__":
    # Define input parameters directly in the script
    input_params = {
        "model_parameters": {
            "bert_model_name": "bert-base-uncased",
            "gpt2_model_name": "gpt2"
        },
        "training_parameters": {
            "learning_rate": 3e-05,
            "batch_size": 16,
            "epochs": 3,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "data": {
            "sentence_pairs": [
                ["He is a talented musician.", "He can play multiple instruments."],
                ["The sun is shining.", "It's a beautiful day outside."],
                ["I love reading books.", "My favorite genre is science fiction."],
                ["The cat is sleeping.", "The dog is barking loudly."],
                ["She went to the grocery store.", "She bought milk and bread."],
                ["The car won't start.", "The battery is dead."],
                ["I'm learning to code.", "I hate programming and computers."],
                ["The movie was exciting.", "I fell asleep halfway through."]
            ],
            "expected_results": [True, True, True, False, True, True, False, False]
        }
    }

    # Load models using parameters from the input_params dictionary
    bert_model, bert_tokenizer = load_bert_model(input_params['model_parameters']['bert_model_name'])
    gpt2_model, gpt2_tokenizer = load_gpt2_model(input_params['model_parameters']['gpt2_model_name'])

    # Set training parameters
    for model in [bert_model, gpt2_model]:
        for param, value in input_params['training_parameters'].items():
            setattr(model, param, value)

    # Get sentence pairs and expected results from input parameters
    sentence_pairs = input_params['data']['sentence_pairs']
    expected_results = input_params['data']['expected_results']

    test_nsp_and_generate(bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer, sentence_pairs, expected_results)