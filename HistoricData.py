import re
import json


def fetch_historic_data(item_id):
    with open('bazaar_history_combined.json', 'r') as f:
        data = json.load(f)
    json_content = str(data)
    buy_data= re.findall(r'"buy":([^,]*),', json_content)
    sell_data = re.findall(r'"sell":([^,]*),', json_content)
    time_data =re.findall(r'"timestamp":"([^"]*)"', json_content)
    buyVolume_data = re.findall(r'"buyVolume":([^,]*),', json_content)
    sellVolume_data = re.findall(r'"sellVolume":([^,]*),', json_content)
    buyMovingWeek_data = re.findall(r'"buyMovingWeek":([^,]*),', json_content)
    sellMovingWeek_data = re.findall(r'"sellMovingWeek":([^},]*)},', json_content)
    maxBuy_data = re.findall(r'"maxBuy":([^,]*),', json_content)
    maxSell_data = re.findall(r'"maxSell":([^,]*),', json_content)
    minBuy_data = re.findall(r'"minBuy":([^,]*),', json_content)
    minSell_data = re.findall(r'"minSell":([^,]*),', json_content)
    years_data = []
    months_data = []
    days_data = []
    for time in time_data:
        year = time[0:4]
        month = time[5:7]
        day = time[8:10]
        years_data .append(year)
        months_data.append(month)
        days_data.append(day)

    years_int = [int(year) for year in years_data]
    months_int = [int(month) for month in months_data]
    days_int = [int(day) for day in days_data]
    buy_data_float = [float(price) for price in buy_data]
    sell_data_float = [float(price) for price in sell_data]
    buyVolume_data_float = [float(volume) for volume in buyVolume_data]
    sellVolume_data_float = [float(volume) for volume in sellVolume_data]
    buyMovingWeek_data_int = [int(moving) for moving in buyMovingWeek_data]
    sellMovingWeek_data_int = [int(moving) for moving in sellMovingWeek_data]
    maxBuy_data_float = [float(maxb) for maxb in maxBuy_data]
    maxSell_data_float = [float(maxs) for maxs in maxSell_data]
    minBuy_data_float = [float(minb) for minb in minBuy_data]
    minSell_data_float = [float(mins) for mins in minSell_data]

    return years_int, months_int, days_int, buy_data_float, sell_data_float, buyVolume_data_float, sellVolume_data_float, buyMovingWeek_data_int, sellMovingWeek_data_int, maxBuy_data_float, maxSell_data_float, minBuy_data_float, minSell_data_float



