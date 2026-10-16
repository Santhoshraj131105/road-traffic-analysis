from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request


app = Flask(__name__)

# Project folders and files
DASHBOARD_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = DASHBOARD_FOLDER.parent

CSV_PATH = PROJECT_FOLDER / "traffic.csv"
MODEL_PATH = DASHBOARD_FOLDER / "traffic_model.joblib"
STATIC_FOLDER = DASHBOARD_FOLDER / "static"
CHART_PATH = STATIC_FOLDER / "forecast.png"

STATIC_FOLDER.mkdir(exist_ok=True)

# Load the trained model and its feature names
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

# These are display labels for junction IDs 1–4.
# The dataset itself does not confirm their real-world locations.
junction_names = {
    1: "Chennai",
    2: "Pondicherry",
    3: "ECR",
    4: "Chengalpattu",
}

# Load and clean the traffic data
data = pd.read_csv(CSV_PATH)
data.columns = data.columns.str.strip()

# Fix a possible spelling variation in the CSV
if "Juction" in data.columns and "Junction" not in data.columns:
    data = data.rename(columns={"Juction": "Junction"})

data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
data["Junction"] = pd.to_numeric(data["Junction"], errors="coerce")
data["Vehicles"] = pd.to_numeric(data["Vehicles"], errors="coerce")

data = data.dropna(
    subset=["DateTime", "Junction", "Vehicles"]
).copy()

data["Junction"] = data["Junction"].astype(int)
data = data.sort_values("DateTime")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    chart_version = None

    selected_junction = request.form.get("junction", "")
    selected_datetime = request.form.get("datetime", "")

    if request.method == "POST":
        try:
            junction_id = int(selected_junction)
            chosen_time = pd.to_datetime(selected_datetime)

            if junction_id not in junction_names:
                raise ValueError("Please select a valid location.")

            # Build the date/time inputs expected by the trained model
            input_values = {column: 0 for column in feature_columns}

            date_features = {
                "Hour": chosen_time.hour,
                "DayOfWeek": chosen_time.dayofweek,
                "Month": chosen_time.month,
                "DayOfMonth": chosen_time.day,
            }

            for column, value in date_features.items():
                if column in input_values:
                    input_values[column] = value

            # Set the selected junction's model input
            junction_column = f"Junction_{junction_id}"

            if junction_column in input_values:
                input_values[junction_column] = 1
            elif "Junction" in input_values:
                input_values["Junction"] = junction_id
            else:
                raise ValueError(
                    "The saved model's features do not match this app."
                )

            model_input = pd.DataFrame(
                [input_values],
                columns=feature_columns
            )

            predicted_vehicles = max(
                0,
                round(float(model.predict(model_input)[0]))
            )

            # Calculate Low, Medium, and High levels from this junction's data
            junction_vehicles = data.loc[
                data["Junction"] == junction_id,
                "Vehicles"
            ]

            if junction_vehicles.empty:
                raise ValueError("No traffic data found for this location.")

            low_limit = junction_vehicles.quantile(1 / 3)
            high_limit = junction_vehicles.quantile(2 / 3)

            if predicted_vehicles < low_limit:
                traffic_level = "Low"
            elif predicted_vehicles < high_limit:
                traffic_level = "Medium"
            else:
                traffic_level = "High"

            # Plot recent historical data and the prediction
            history = (
                data[data["Junction"] == junction_id]
                .sort_values("DateTime")
            )

            recent = history[
                history["DateTime"] <= chosen_time
            ].tail(48)

            if recent.empty:
                recent = history.tail(48)

            plt.figure(figsize=(10, 5))
            plt.plot(
                recent["DateTime"],
                recent["Vehicles"],
                marker="o",
                markersize=3,
                label="Historical vehicles"
            )
            plt.scatter(
                [chosen_time],
                [predicted_vehicles],
                color="red",
                s=100,
                label="Predicted vehicles"
            )
            plt.title(f"Traffic prediction — {junction_names[junction_id]}")
            plt.xlabel("Date and time")
            plt.ylabel("Vehicles per hour")
            plt.xticks(rotation=30)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(CHART_PATH)
            plt.close()

            result = {
                "location": junction_names[junction_id],
                "time": chosen_time.strftime("%d %b %Y, %I:%M %p"),
                "vehicles": predicted_vehicles,
                "level": traffic_level,
            }
            chart_version = pd.Timestamp.now().timestamp()

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        locations=sorted(junction_names.items()),
        result=result,
        error=error,
        selected_junction=selected_junction,
        selected_datetime=selected_datetime,
        chart_version=chart_version,
    )


if __name__ == "__main__":
    app.run(debug=True)