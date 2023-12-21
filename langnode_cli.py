from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

model.eval()  # Set the model to inference mode


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Interact with a language model")
    parser.add_argument('text', type=str, help='Text to input into the model')
    return parser.parse_args()

def main():
    args = get_args()
    input_text = args.text

    # Process the input and get a response from the model
    response = interact_with_model(input_text)
    print(response)

def interact_with_model(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    main()
