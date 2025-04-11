from flask import Flask, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/generate", methods=["GET"])
def generate_quiz():
    prompt = "次の形式で中学生向けのクイズを1問作ってください：\n問題：\n選択肢：\nA. \nB. \nC. \nD. \n正解：\n解説："
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, do_sample=True, temperature=0.8)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"quiz": result})
