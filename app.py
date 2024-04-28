from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

rfc = joblib.load("trained_model.joblib")

categorical_mapping = {
    'Industry': {'Finance': 0, 'Healthcare': 1, 'Retail': 2, 'Technology': 3},
    'Sentiment': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
    'Completion Status': {'Completed': 0, 'Incomplete': 1},
    'Customer Size': {'Medium': 0, 'Small': 1, 'Large': 2},
    'Lead Status': {'Negotiation': 0, 'New': 1, 'Contacted': 2, 'Closed-Won': 3, 'Closed-Lost': 4, 'Opportunity': 5, 'Qualified': 6, 'Engaged': 7}
}

def preprocess_input(input_data):
    for column, mapping in categorical_mapping.items():
        input_data[column] = mapping.get(input_data[column], input_data[column])
    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    preprocessed_data = preprocess_input(input_data)
    df = pd.DataFrame.from_dict([preprocessed_data])
    ypred = rfc.predict(df)
    predicted_risk_labels = "".join(ypred)
    return jsonify({"predicted_risk_labels": predicted_risk_labels})

if __name__ == '__main__':
    app.run(debug=True)
