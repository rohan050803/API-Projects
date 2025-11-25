# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face tools for loading the model and tokenizer
import gradio as gr  # Gradio is used for building a simple web-based UI

# ===============================
# Load the Flan-T5 model and tokenizer from Hugging Face
# ===============================
model_name = "google/flan-t5-large"  # Pretrained model name
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Loads the tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Loads the model

# Set the model to evaluation mode (important for inference)
model.eval()

# ===============================
# Function to generate a response using the model
# ===============================
def generate_response(input_text, context=""):
    # Add explicit instruction for conversation
    prompt = f"You are a helpful assistant. Continue the conversation below.\n"
    if context:
        prompt += f"{context}\n"
    prompt += f"User: {input_text}\nAssistant:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,          # only generate 150 tokens for the response
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Strip the prompt from the output if model echoes the instruction
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response


# ===============================
# Function to handle multi-turn chat logic
# ===============================
def chat(user_input, history=None):
    if history is None:
        history = []  # Initialize history if it's the first message

    # Build the conversation context from previous turns
    context = ""
    for turn in history:
        context += f"User: {turn[0]}\nAssistant: {turn[1]}\n"

    # Generate a new response based on current input and past context
    response = generate_response(user_input, context)

    # Add the new turn to the conversation history
    history.append((user_input, response))

    # Return updated history for display and storage
    return history, history

# ===============================
# Build the Gradio web UI
# ===============================
with gr.Blocks() as app:
    gr.Markdown("##  Flan-T5 Chatbot")  # Title/heading in the web app

    # Creates a chat window to display the conversation.
    # It will show user messages and bot responses in a chat-like format.
    chatbot = gr.Chatbot()  

    msg = gr.Textbox(label="Your Message", placeholder="Type your message and press Enter")  # User input box

    state = gr.State([])  # holds persistent data during the session., Holds the chat history (list of messages)

    # When user submits a message:
    # Calls the function chat with inputs [msg, state].
    # Updates outputs [chatbot, state].
    submit_btn = msg.submit(chat, [msg, state], [chatbot, state])  # Call 'chat' function and update the chatbot and state

    # Clear the input box after submission
    submit_btn.then(lambda: "", None, msg)
    # After submission, this clears the Textbox so the user can type a new message.
    # lambda: "" returns an empty string.
    # None → no input needed for this step.
    # msg → updates the Textbox component.
# ===============================
# Launch the web application
# ===============================
app.launch()
