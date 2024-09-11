import io
import os
import json
import uuid
import datetime

import pandas as pd

from flask import Flask, render_template, request, session, abort
from flask_bootstrap import Bootstrap5
from flask_sslify import SSLify

from google.cloud import storage

import itertools
from sitepackages.eloRating import EloSystem

#setup the app
app = Flask(__name__)
bootstrap = Bootstrap5(app)
sslify = SSLify(app)

with open('creds.json') as f:
    creds = json.load(f)
app.secret_key = creds["secret_key"]

# set gcloud environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-creds.json"

storage_client = storage.Client()

tier_dict = {"High": 3, "Medium": 2, "Low": 1}

def initTempBucket(bucket, days= 3):
    """Set the lifecycle rules for the bucket"""
    if not bucket.exists():
        bucket.create()
    else:
        bucket = storage_client.get_bucket(session["id"])
    bucket.add_lifecycle_delete_rule(age=days)
    bucket.patch()

@app.before_request
def before_request():
    session.permanent = True
    if "id" not in session:
        session["id"] = uuid.uuid4().hex

@app.route("/", methods= ["GET"])
def index():
    return render_template("index.html")

@app.route("/app", methods= ["GET", "POST"])
def applet():
    bucket = storage_client.bucket(session["id"])
    initTempBucket(bucket)
    if request.files:
        for file in request.files.getlist('formFileMultiple'):
            file_name = file.filename
            file_content = file.stream.read()
            bucket_file_name = uuid.uuid4().hex + "_" + file_name
            blob = bucket.blob(bucket_file_name)
            df = pd.read_excel(io.BytesIO(file_content))
            metrics = list(df.columns)[1:]
            blob.metadata = {"selected_metric": "", "metrics": json.dumps(metrics), "paper_confidence": "", "usecase_confidence": "", "file_name": file_name, "bucket_file_name": bucket_file_name}
            blob.upload_from_file(io.BytesIO(file_content))   # retrieve users files
    userfilestoragelist = []
    blobs = bucket.list_blobs()
    for blob in blobs:
        blob_metadata = blob.metadata
        blob_metadata["metrics"] = json.loads(blob_metadata["metrics"])
        blob_metadata["json"] = json.dumps(blob_metadata)
        userfilestoragelist.append(blob_metadata)
    return render_template("app.html", userfilestorage= userfilestoragelist)

@app.route("/api/updatemetadata", methods= ["POST"])
def addfilemetadata():
    if request.form["action"] == "update":
        file_metadata = json.loads(request.form["file_metadata"])
        bucket = storage_client.get_bucket(session["id"])
        blob = bucket.blob(file_metadata["bucket_file_name"])
        if "metrics" in file_metadata:
            file_metadata["metrics"] = json.dumps(file_metadata["metrics"])
        blob.metadata = file_metadata
        blob.patch()
        return file_metadata
    else:
        abort(422)

@app.route("/api/delete", methods= ["POST"])
def deletefile():
    if request.form["action"] == "delete":
        file_metadata = json.loads(request.form["file_metadata"])
        bucket = storage_client.get_bucket(session["id"])
        blob = bucket.blob(file_metadata["bucket_file_name"])
        blob.delete()
        return file_metadata["bucket_file_name"]
    else:
        abort(422)

@app.route("/api/calculate", methods= ["POST"])
def calculateElo():
    if request.form["action"] == "calculate" and len(json.loads(request.form["files"])) > 1:
        performance_dfs = []
        algos = {}
        failed_files = []
        for file in json.loads(request.form["files"]):
            try:
                bucket = storage_client.get_bucket(session["id"])
                blob = bucket.get_blob(file)
                temp_df = pd.read_excel(io.BytesIO(blob.download_as_bytes()))
                temp_metadata = blob.metadata
                temp_df = temp_df.set_index(temp_df.columns[0])
                temp_algos = temp_df.index
                for algo in temp_algos:
                    if algo not in algos.keys():
                        algos[algo] = [temp_metadata["file_name"]]
                    else:
                        algos[algo].append(temp_metadata["file_name"])
                temp_df = temp_df.astype(float)
                temp_df_dict = temp_df[temp_metadata["selected_metric"]].to_dict()
                temp_df_dict = itertools.combinations([(key, temp_df_dict[key]) for key in temp_df_dict], r=2)
                temp_df_dict = [(comp[0][0], comp[1][0], comp[0][0]) if comp[0][1] > comp[1][1] else (comp[0][0], comp[1][0], None)
                if comp[0][1] >= comp[1][1] else (comp[0][0], comp[1][0], comp[1][0]) for comp in temp_df_dict]
                if temp_metadata["paper_confidence"]:
                    paper_confidence = temp_metadata["paper_confidence"]
                else:
                    paper_confidence = "Medium"
                if temp_metadata["usecase_confidence"]:
                    usecase_confidence = temp_metadata["usecase_confidence"]
                else:
                    usecase_confidence = "Medium"
                temp_df_dict = {"file": temp_metadata["file_name"],"paper_confidence": tier_dict[paper_confidence], "usecase_confidence": tier_dict[usecase_confidence], "data": temp_df_dict, "algorithms": temp_algos}
                performance_dfs.append(temp_df_dict)
            except:
                failed_files.append(file)

        cv_ranking = pd.DataFrame()
        for fold in range(5):
            elo = EloSystem(use_mov= True, mov_delta= 2)
            for algorithm in algos.keys():
                elo.add_player(algorithm)
            for performance_df in performance_dfs:
                for comp in performance_df["data"]:
                    elo.record_match(*comp, mov= (performance_df["paper_confidence"] + performance_df["usecase_confidence"])/2)
            ranking = elo.get_overall_list()
            for i in range(len(ranking)):
                ranking[i]["prob"] = ranking[i]["prob"]
                ranking[i]["files"] = str(algos[ranking[i]["player"]]).replace("[", "").replace("]", "").replace("'", "").replace(",", ", ")
            cv_ranking = pd.concat((cv_ranking, pd.DataFrame(ranking)))
        cv_ranking = cv_ranking.groupby(["player", "files"]).mean().reset_index()
        cv_ranking["prob"] = cv_ranking["prob"].apply(lambda x: "{0:.0%}".format(x))
        cv_ranking = cv_ranking.sort_values("elo", ascending=False)
        cv_ranking["rank"] = cv_ranking.rank(ascending=False)["elo"]
        cv_ranking = cv_ranking.rename(columns= {"player": "algorithm"})
        file_name = uuid.uuid4().hex + ".xlsx"
        temp_xlsx = "/tmp/" + file_name
        pd.DataFrame(cv_ranking).to_excel(temp_xlsx)
        bucket = storage_client.bucket(session["id"] + "_results")
        initTempBucket(bucket)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(temp_xlsx)
        file_url = blob.generate_signed_url(datetime.timedelta(seconds= 600), method= "GET")
        os.remove(temp_xlsx)
        return render_template("results.html", results= cv_ranking.to_dict('records'),
                               ranking_file= file_url, failed_files= failed_files)
    else:
        abort(422)

if __name__ == "__main__":
    app.config.update(SESSION_COOKIE_SECURE=True)
    app.run(debug= False)