from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
models = {
    'SVM': {'path': os.path.join(current_dir, 'model', 'svm_model.pkl'), 'accuracy': 89},
}
vector_path = os.path.join(current_dir, 'model', 'vectorizer.pkl')
vector = joblib.load(vector_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    accuracy = None
    user_input = request.form.get('user_input', '')

    if request.method == 'POST' and user_input:
        try:
            model_path = models['SVM']['path']
            model = joblib.load(model_path)
            accuracy = models['SVM']['accuracy']
            
            input_tf_idf = vector.transform([user_input])
            raw_prediction = model.predict(input_tf_idf)[0]
            prediction = "Positive" if raw_prediction == 1 else "Negative"
            
        except Exception as e:
            print(f"Error: {e}")

    return render_template('index.html', prediction=prediction, accuracy=accuracy, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)


