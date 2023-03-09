from flask import Flask, jsonify, request
import chatbot

app = Flask(__name__)

bot = chatbot.FAQ_chatbot()

print("Chatbot is ready!!!")


@app.route('/bot', methods=["POST"])
def get_answer():
    if request.method == "POST":
        question = request.form.get("Question")
        answer = bot.get_answer(question)

        return jsonify(answer)


app.run()

