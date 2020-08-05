import os
import flask

import predict

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template("index.html")

@app.route("/uploader", methods = ["POST"])
def upload_file():
    trace = flask.request.files["trace"]
    with open("trace_temp.json", "wb") as f:
        f.write(trace.read())
    model = flask.request.files["model"]
    with open("model_temp.mdl", "wb") as f:
        f.write(model.read())
    
    typeClass = predict.Prediction("trace_temp.json", "model_temp.mdl")
    os.remove("trace_temp.json")
    os.remove("model_temp.mdl")
    return flask.render_template("prediction.html", typeClass = typeClass)

if __name__ == "__main__":
    app.run(debug=True)