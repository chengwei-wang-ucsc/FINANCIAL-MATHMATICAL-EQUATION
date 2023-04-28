import numpy as np
import pandas as pd

def Upper_shadow_5(df):
    n = 5+1
    high = df['high']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - np.maximum(close.iloc[i-n:i],open_.iloc[i-n:i])
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Upper_shadow_13(df):
    n = 13+1
    high = df['high']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - np.maximum(close.iloc[i-n:i],open_.iloc[i-n:i])
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Upper_shadow_21(df):
    n = 21+1
    high = df['high']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - np.maximum(close.iloc[i-n:i],open_.iloc[i-n:i])
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Upper_shadow_34(df):
    n = 34+1
    high = df['high']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - np.maximum(close.iloc[i-n:i],open_.iloc[i-n:i])
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Upper_shadow_55(df):
    n = 55+1
    high = df['high']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - np.maximum(close.iloc[i-n:i],open_.iloc[i-n:i])
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Upper_shadow_89(df):
    n = 89+1
    high = df['high']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - np.maximum(close.iloc[i-n:i],open_.iloc[i-n:i])
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Lower_shadow_5(df):
    n = 5+1
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        lowerShadow = np.minimum(close.iloc[i-n:i], open_.iloc[i-n:i]) - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(n-1).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Lower_shadow_13(df):
    n = 13+1
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        lowerShadow = np.minimum(close.iloc[i-n:i], open_.iloc[i-n:i]) - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(n-1).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Lower_shadow_21(df):
    n = 21+1
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        lowerShadow = np.minimum(close.iloc[i-n:i], open_.iloc[i-n:i]) - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(n-1).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Lower_shadow_34(df):
    n = 34+1
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        lowerShadow = np.minimum(close.iloc[i-n:i], open_.iloc[i-n:i]) - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(n-1).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Lower_shadow_55(df):
    n = 55+1
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        lowerShadow = np.minimum(close.iloc[i-n:i], open_.iloc[i-n:i]) - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(n-1).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Lower_shadow_89(df):
    n = 89+1
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    res = [0]*n
    for i in range(n, len(df)):
        lowerShadow = np.minimum(close.iloc[i-n:i], open_.iloc[i-n:i]) - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(n-1).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Williams_upper_shadow_6(df):
    n = 6+1
    high = df['high']
    close = df['close']
    res = [0]*n        
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - close.iloc[i-n:i]
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Williams_upper_shadow_13(df):
    n = 13+1
    high = df['high']
    close = df['close']
    res = [0]*n        
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - close.iloc[i-n:i]
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Williams_upper_shadow_21(df):
    n = 21+1
    high = df['high']
    close = df['close']
    res = [0]*n        
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - close.iloc[i-n:i]
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Williams_upper_shadow_34(df):
    n = 34+1
    high = df['high']
    close = df['close']
    res = [0]*n        
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - close.iloc[i-n:i]
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Williams_upper_shadow_55(df):
    n = 55+1
    high = df['high']
    close = df['close']
    res = [0]*n        
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - close.iloc[i-n:i]
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Williams_upper_shadow_89(df):
    n = 89+1
    high = df['high']
    close = df['close']
    res = [0]*n        
    for i in range(n, len(df)):
        upperShadow = high.iloc[i-n:i] - close.iloc[i-n:i]
        std_upperShadow = upperShadow / upperShadow.rolling(n-1).mean().shift(1)
        res.append(std_upperShadow.iloc[-1])
    return res

def Williams_lower_shadow_6(df):
    n = 6+1
    close = df['close']
    low = df['low']
    res = [0]*n    
    for i in range(n, len(df)):
        lowerShadow = close.iloc[i-n:i] - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(5).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Williams_lower_shadow_13(df):
    n = 13+1
    close = df['close']
    low = df['low']
    res = [0]*n    
    for i in range(n, len(df)):
        lowerShadow = close.iloc[i-n:i] - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(5).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Williams_lower_shadow_21(df):
    n = 21+1
    close = df['close']
    low = df['low']
    res = [0]*n    
    for i in range(n, len(df)):
        lowerShadow = close.iloc[i-n:i] - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(5).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Williams_lower_shadow_34(df):
    n = 34+1
    close = df['close']
    low = df['low']
    res = [0]*n    
    for i in range(n, len(df)):
        lowerShadow = close.iloc[i-n:i] - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(5).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Williams_lower_shadow_55(df):
    n = 55+1
    close = df['close']
    low = df['low']
    res = [0]*n    
    for i in range(n, len(df)):
        lowerShadow = close.iloc[i-n:i] - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(5).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def Williams_lower_shadow_89(df):
    n = 89+1
    close = df['close']
    low = df['low']
    res = [0]*n    
    for i in range(n, len(df)):
        lowerShadow = close.iloc[i-n:i] - low.iloc[i-n:i]
        std_lowerShadow = lowerShadow / lowerShadow.rolling(5).mean().shift(1)
        res.append(std_lowerShadow.iloc[-1])
    return res

def TrendStrength(df):
    #计算路程之和
    distance = df.close - df.close.shift(1)
    distance = distance.dropna(axis=0)
    distance_total = distance.apply(abs)
    distance_total = distance_total.apply(sum)
    #计算位移
    displacement = df.close.iloc[-1] - df.close.iloc[0]
    #计算因子值
    trend_strength = pd.DataFrame()
    trend_strength['TrendStrength'] = displacement / distance_total
    return trend_strength