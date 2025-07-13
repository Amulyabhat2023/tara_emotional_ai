from flask import Flask, render_template, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load Tara's AI Brain - Using a specialized emotional support model
print("Loading Tara's brain... Please wait.")
model_name = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
print("Tara is ready to chat! Enter quit to exit")

# 3. Create the route for the main webpage
@app.route('/')
def home():
    return render_template('index.html')

# 4. Enhanced chatbot endpoint with better conversation handling
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    history = data.get('history', [])
    
    # Format conversation history with speaker labels
    formatted_history = "\n".join(
        [f"User: {text}" if i % 2 == 0 else f"Assistant: {text}" 
         for i, text in enumerate(history)]
    )
    if user_message.lower() == 'quit':
        print("Tara: It was nice talking to you. Take care!")

    # Create the full prompt with context
    prompt = (
        "You are Tara, an empathetic AI companion who provides emotional support. "
        "Respond with warmth, understanding, and encouragement.\n\n"
        f"User: {user_message}\n"
        "Assistant:"
    )
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response with adjusted parameters
    output = model.generate(**inputs)
    
    # Decode and clean the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'reply': response_text})

# 5. Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
