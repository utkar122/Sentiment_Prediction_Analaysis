from flask import Flask, request, render_template
from pickleDemo import SentimentRecommender
import pandas as pd

app = Flask(__name__)

sent_reco_model = SentimentRecommender()

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the username as input
    user_name_input = request.form['username'].lower()
    sent_reco_output = sent_reco_model.top5_recommendations(user_name_input)

    if sent_reco_output is not None:  # Check if data exists
        print("*"*30)
        return render_template("index.html", output=sent_reco_output)
    else:
        message = "The user '{}' does not exist. Please provide a valid username.".format(user_name_input)
        print("$"*30)
        return render_template("welcome.html", output=pd.DataFrame(),message_display=message)
    
if __name__ == '__main__':
    app.run()
