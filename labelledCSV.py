import csv
from typing import List

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import collections

INPUT_SIZE = 300
# FILL YOUR OWN MT5 ACCOUNT INFORMATION
ACCOUNT = None
PASSWORD = None
SERVER = None


def connectMT5(account: int, password: str, server: str):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    authorized = mt5.login(
        account, password=password, server=server)
    if authorized:
        print("connected to account #{}".format(account))
    else:
        print(f'failed to connect at account #{account}, '
              f'error code: {mt5.last_error()}')
        mt5.shutdown()
        quit()


def savePrices(priceFrame: pd.DataFrame, fileName: str):
    val = priceFrame.drop(range(17 + INPUT_SIZE - 1))
    val.to_csv(fileName, index=False)


def findAction(rates: List, index: int):
    openPrice = rates[index][3]
    buy = openPrice + 0.002
    sell = openPrice - 0.002
    for i in range(index + 1, index + 40):
        if rates[i][1] >= buy and rates[i][2] <= sell:
            return 0
        elif rates[i][1] >= buy:
            return 1
        elif rates[i][2] <= sell:
            return 2
    return 0


def makeData(account: int, password: str, server: str, start: int, num: int):
    connectMT5(account, password, server)
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, start, num)
    mt5.shutdown()

    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame.drop(columns=["time", "tick_volume", "real_volume", "spread"],
                     inplace=True)

    savePrices(rates_frame, "prices.csv")

    rates_frame.ta.sma(length=10, append=True)
    rates_frame.ta.rsi(append=True)
    rates_frame.ta.stoch(append=True)
    rates_frame.drop(range(17), inplace=True)

    rates_list = rates_frame.values.tolist()
    to_write = collections.deque(maxlen=INPUT_SIZE*8)
    start = 0
    with open('labelledTest.csv', 'w', newline='') as FILE:
        writer = csv.writer(FILE)
        for i in range(len(rates_list) - 40):
            start += 1
            to_write.extend(rates_list[i])
            if start >= INPUT_SIZE:
                final = list(to_write)
                final.append(findAction(rates_list, i))
                writer.writerow(final)


if __name__ == "__main__":
    makeData(ACCOUNT, PASSWORD, SERVER, 1, 30000)

    # read = pd.read_csv("prices.csv")
    # print(read)
