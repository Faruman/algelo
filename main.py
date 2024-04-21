import json

from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_sslify import SSLify

from sitepackages.eloRating import EloSystem

#setup the app
app = Flask(__name__)
bootstrap = Bootstrap5(app)
#sslify = SSLify(app)

with open('creds.json') as f:
    creds = json.load(f)
app.secret_key = creds["secret_key"]

@app.route("/", methods= ["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.config.update(SESSION_COOKIE_SECURE=True)
    app.run(debug= False)