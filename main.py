from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import fileForDoc

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Date_of_Journey
        date_dep = request.form["Date_of_Journey"]
        Day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)
        year = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").year)

        # print("Journey Date : ",Journey_day, Journey_month)

        # Departure
        dep_time = request.form["Dep_Time"]
        Dep_Time = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M"))

        # Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        # Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_Time = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M"))
        # Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        # Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        # dur_hour = abs(Arrival_hour - Dep_hour)
        # dur_min = abs(Arrival_min - Dep_min)
        Duration = abs(int(pd.to_datetime(date_arr - dep_time)))
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        # AIR ASIA = 0 (not in column)
        Airline = request.form['airline']

        # Source
        # Banglore = 0 (not in column)
        Source = request.form["Source"]

        Destination = request.form["Destination"]

        output = round(fileForDoc.prediction[0], 2)

        return render_template('index.html', prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8030, debug=True)
	#app.run(debug=True) # running the app