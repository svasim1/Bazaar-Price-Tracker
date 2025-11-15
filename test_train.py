import json
import pickle
import re
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

MODEL_PATH = "trained_model.pkl"
ITEM_FILE = "bazaar_history_combined_BOOSTER_COOKIE.json"

with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

with open(ITEM_FILE,"r") as f:
    data = json.load(f)

json_text = str(data)

timestamps  = re.findall(r'"timestamp":"([^"]*)"', json_text)
buy_price   = re.findall(r'"buy":([^,]*),', json_text)
sell_price  = re.findall(r'"sell":([^,]*),', json_text)
buy_vol     = re.findall(r'"buyVolume":([^,]*),', json_text)
sell_vol    = re.findall(r'"sellVolume":([^,]*),', json_text)
buy_week    = re.findall(r'"buyMovingWeek":([^,]*),', json_text)
sell_week   = re.findall(r'"sellMovingWeek":([^},]*)}', json_text)
max_buy     = re.findall(r'"maxBuy":([^,]*),', json_text)
max_sell    = re.findall(r'"maxSell":([^,]*),', json_text)
min_buy     = re.findall(r'"minBuy":([^,]*),', json_text)
min_sell    = re.findall(r'"minSell":([^,]*),', json_text)

X = []
y_true = []
ts_dt = []

for i in range(len(timestamps)):
    try:
        x = [
            float(buy_price[i]),
            float(sell_price[i]),
            float(buy_vol[i]),
            float(sell_vol[i]),
            float(buy_week[i]),
            float(sell_week[i]),
            float(max_buy[i]),
            float(max_sell[i]),
            float(min_buy[i]),
            float(min_sell[i]),
        ]
        X.append(x)
        y_true.append(float(sell_price[i]))
        ts_dt.append(datetime.fromisoformat(timestamps[i].replace("Z","")))
    except:
        pass

X = np.array(X)
preds_hist = model.predict(X)

last_feature = X[-1]
future_preds = []
future_ts = []

today = datetime.now()
for i in range(21):  # 3 weeks
    future_ts.append(today + timedelta(days=i))
    future_preds.append(model.predict([last_feature])[0])

plt.figure(figsize=(16,6))
plt.plot(ts_dt, y_true, label="Historical True")
plt.plot(ts_dt, preds_hist, label="Historical Prediction")
plt.plot(future_ts, future_preds, label="Future Forecast", linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()