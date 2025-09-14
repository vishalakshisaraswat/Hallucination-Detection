from flask import Flask, render_template, request
from pipeline import check_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        user_text = request.form["user_text"]
        results = check_text(user_text)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
