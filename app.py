from flask import Flask, render_template, request, jsonify
from chatbot_engine import chatbot

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_msg = request.form.get("msg")
    response = chatbot("user1", user_msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host = "0.0.0.0" , port=10000)
