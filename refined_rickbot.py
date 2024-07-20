from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Alpaca prompt template
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

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

from transformers import TextStreamer





# # Initialize conversation history
# conversation_history = []

# def format_conversation(history):
#     # Concatenate the conversation history with prompts and responses
#     formatted_history = ""
#     for i, (prompt, response) in enumerate(history):
#         formatted_history += alpaca_prompt.format(prompt, response)
#     return formatted_history

# def chat(model, tokenizer, max_history=5):
#     # Initialize TextStreamer for output streaming
#     text_streamer = TextStreamer(tokenizer)

#     while True:
#         # Get user input
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
        
#         # Append the user input to the conversation history
#         conversation_history.append((user_input, ""))
        
#         # Keep only the last `max_history` turns
#         conversation_history[:] = conversation_history[-max_history:]

#         # Format the conversation history for the model
#         formatted_history = format_conversation(conversation_history)
        
#         # Tokenize the formatted history
#         inputs = tokenizer([formatted_history], return_tensors="pt").to("cuda")
        
#         # Generate a response
#         outputs = model.generate(**inputs, max_new_tokens=128)
        
#         # Decode the generated response
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Extract only the bot's response after the last prompt
#         bot_response = response.split(alpaca_prompt.format(user_input, ""))[-1].strip()
        
#         # Update the last user input with the model response in the conversation history
#         conversation_history[-1] = (user_input, bot_response)
        
#         # Print the model response
#         print(f"Bot: {bot_response}")

# # Example usage
# chat(model, tokenizer)
