import requests

data = {
    "features": {
        "OPERA_Latin American Wings": 1,
        "MES_7": 0,
        "MES_10": 0,
        "OPERA_Grupo LATAM": 1,
        "MES_12": 0,
        "TIPOVUELO_I": 1,
        "MES_4": 1,
        "MES_11": 0,
        "OPERA_Sky Airline": 0,
        "OPERA_Copa Air": 0
    }
}

try:
    response = requests.post("https://postpredict-2i3kxvjh2a-rj.a.run.app/predict", json=data)

    if response.status_code == 200:
        prediction_data = response.json()
        print("Prediction:", prediction_data.get("prediction"))
    else:
        print("Error:", response.status_code)
except requests.exceptions.RequestException as e:
    print("Error:", e)