import math 
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import os

EIA_KEY = "xXYGzGQnZH5eNsc8QdRTKPUSN1k5vOjVr4szBuSZ"
LOG_PATH = "agent_actions.csv"

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

def get_caiso_demand():
    url = (
        "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        "?frequency=hourly"
        "&data[0]=value"
        "&facets[respondent][]=CAL"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&length=1"
        f"&api_key={EIA_KEY}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception("EIA API failed:", r.text)
    response = r.json().get('response', {}).get('data', [])
    if not response:
        raise Exception("EIA API returned no demand data.")
    latest = response[0]
    hour = datetime.strptime(latest['period'], "%Y-%m-%dT%H").hour
    return float(latest['value']), hour

def get_temp(lat=34.05, lon=-118.25):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m"
        )
        response = requests.get(url, timeout=10)
        data = response.json()
        if "current" in data and "temperature_2m" in data["current"]:
            return float(data["current"]["temperature_2m"])
        else:
            raise ValueError("Temperature data not found.")
    except Exception as e:
        current_hour = datetime.now().hour
        return 18 + 5 * math.sin((current_hour - 6) * math.pi / 12)

def normalize(x, mean, std):
    return (x - mean) / std

def run_agent():
    capacity = 10
    soc = 5
    charge_rate = 2
    discharge_rate = 2

    state_size = 4
    action_size = 3
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load("dqn_model.pt"))
    model.eval()

    demand, hour = get_caiso_demand()
    temp = get_temp()
    price = 30 + 2 * temp

    state = torch.FloatTensor([
        soc / capacity,
        normalize(demand, 20000, 4000),
        normalize(price, 100, 20),
        hour / 24
    ])

    with torch.no_grad():
        q = model(state)
        action = torch.argmax(q).item()

    if action == 1 and soc < capacity:
        soc += charge_rate
    elif action == 2 and soc > 0:
        soc -= discharge_rate
    soc = max(0, min(capacity, soc))

    action_str = ["Hold", "Charge", "Discharge"][action]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Timestamp": now,
        "Hour": hour,
        "Demand": demand,
        "Temp_C": temp,
        "Price": price,
        "Action": action_str,
        "SOC": soc
    }

    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)

    return row

# Minimal Gradio wrapper
import gradio as gr

def gradio_run():
    result = run_agent()
    return f"{result['Timestamp']} | Action: {result['Action']} | SOC: {result['SOC']} MWh"

demo = gr.Interface(fn=gradio_run, inputs=[], outputs="text")
if __name__ == "__main__":
    demo.launch()
