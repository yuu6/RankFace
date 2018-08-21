# -*- coding:utf-8 -*-
"""
@Time:2018/8/21 11:32
@Author:yuhongchao
"""
import pandas as pd
import numpy as np

if __name__ == "__main__":
	d = pd.read_excel("Attractiveness label.xlsx", sheet_name="Sheet1")
	df = pd.DataFrame(d.values, columns=["index", "rate", "dev"])
	df.drop(axis=0, columns=["dev"], inplace=True)
	df['rate'] = df['rate'] * 20
	df = df.astype("int", copy=False)
# print(pd)
# print(np.max(pd["rate"]),np.argmax(pd['rate']))
	pd.to_pickle(df,"../labels.pickle")
