import torch
from flask import Flask, request, jsonify, render_template
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True 

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Respond to the given text the way Rick would in the show Rick and Morty using as much context from the input to harbor a response

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Function to format the prompts
def formatting_prompts_func(examples):
    inputs = examples["Input"]
    outputs = examples["Output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts,}

from transformers import TextStreamer

app = Flask(__name__)

conversation_history = []

def format_conversation(history):
    # Concatenate the conversation history with prompts and responses
    formatted_history = ""
    for i, (prompt, response) in enumerate(history):
        formatted_history += alpaca_prompt.format(prompt, response)
    return formatted_history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    global conversation_history
    conversation_history.append((user_input, ""))

    # Keep only the last 5 turns of conversation history
    conversation_history[:] = conversation_history[-5:]

    # Format the conversation history for the model
    formatted_history = format_conversation(conversation_history)

    # Tokenize the formatted history
    inputs = tokenizer([formatted_history], return_tensors="pt").to("cuda")

    # Generate a response
    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the bot's response after the last prompt
    bot_response = response.split(alpaca_prompt.format(user_input, ""))[-1].strip()

    # Update the last user input with the model response in the conversation history
    conversation_history[-1] = (user_input, bot_response)

    return jsonify({"user_input": user_input, "bot_response": bot_response, "history": conversation_history})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')