# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "secret-key"

# EXCELからデータを読み込む
vectors = {
    "FW": pd.read_excel("プレミア診断.xlsx", sheet_name="FWベクトル"),
    "MF": pd.read_excel("プレミア診断.xlsx", sheet_name="MFベクトル"),
    "DF": pd.read_excel("プレミア診断.xlsx", sheet_name="DFベクトル"),
    "GK": pd.read_excel("プレミア診断.xlsx", sheet_name="GKベクトル"),
}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/form", methods=["GET"])
def form():
    position = request.args.get("position")
    if position not in vectors:
        return redirect(url_for("home"))
    session["position"] = position  # 選んだポジションを保存
    return render_template("index.html")


@app.route("/transition")
def transition():
    position = request.args.get("position", "")
    return render_template("transition.html", position=position)


@app.route("/match", methods=["POST"])
def match_player():
    data = request.json
    user_input = np.array(data["input"], dtype=float).reshape(1, -1)
    position = session.get("position", "FW")
    df = vectors[position]
    player_names = df["名前"].tolist()
    player_vectors = df.drop(columns=["名前"]).values

    sims = cosine_similarity(user_input, player_vectors)[0]
    top_idx = np.argmax(sims)
    return jsonify(
        {"name": player_names[top_idx], "score": round(sims[top_idx] * 100, 2)}
    )


@app.route("/result")
def result():
    name = request.args.get("name", "")
    score = request.args.get("score", "")
    return render_template("result.html", name=name, score=score)


if __name__ == "__main__":
    app.run(debug=True)
