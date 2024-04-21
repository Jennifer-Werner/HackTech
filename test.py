from flask import Flask, render_template
from transformers import BioGptModel, BioGptConfig

app = Flask(__name__)

@app.route('/')
def index():
    # Initializing a BioGPT microsoft/biogpt style configuration
    configuration = BioGptConfig()

    # Initializing a model from the microsoft/biogpt style configuration
    model = BioGptModel(configuration)

    # Accessing the model configuration
    configuration = model.config

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
