import chatbot
# from chatbot import FAQ_chatbot

bot = chatbot.FAQ_chatbot()
question = "How Many People Live Without Health Insurance? "
answer = bot.get_answer(question)
print(answer)

# from flask import Flask, request, jsonify
#
# app = Flask(__name__)
#
#
# @app.route('/', methods=["POST"])
# def home():
#     if request.method == "POST":
#         question = request.form.get("Question")
#         print(question)
#         return jsonify(question)
#
#
# app.run()
