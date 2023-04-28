from scipy.stats import rankdata
import scipy as sp
import numpy as np
import pandas as pd
import statsmodels.api as sm

def Log(sr):
    #自然对数函数
    return np.log(sr)

def Rank(sr):
    #列-升序排序并转化成百分比
    return sr.rank(pct=True)

def Delta(sr,period):
    #period日差分
    return sr.diff(period)

def Delay(sr,period):
    #period阶滞后项
    return sr.shift(period)

def Corr(x,y,window):
    #window日滚动相关系数
    return x.rolling(window).corr(y)

def Cov(x,y,window):
    #window日滚动协方差
    return x.rolling(window).cov(y)

def Sum(sr,window):
    #window日滚动求和
    return sr.rolling(window).sum()

def Prod(sr,window):
    #window日滚动求乘积
    #return df.rolling(window).apply(lambda x: np.prod(x))
    sr1 = sr.copy()
    for i in range(window-1):
        sr1 = sr1*sr.shift(i+1)
    return sr1    

def Mean(sr,window):
    #window日滚动求均值
    return sr.rolling(window).mean()

def Std(sr,window):
    #window日滚动求标准差
    return sr.rolling(window).std()

def Tsrank(sr, window):
    #window日序列末尾值的顺位
    #return df.rolling(window).apply(lambda x: rankdata(x)[-1])
    sr1 = sr.copy()
    sr1[0:window] = np.nan
    for i in range(window,len(sr)+1):
        sr1.iloc[i-1:i] = sr.iloc[i-window:i].rank().iloc[-1]
    return sr1
               
def Tsmax(sr, window):
    #window日滚动求最大值    
    return sr.rolling(window).max()

def Tsmin(sr, window):
    #window日滚动求最小值    
    return sr.rolling(window).min()

def Sign(sr):
    #符号函数
    df = sr.to_frame()
    df1 = df.copy()
    df1[df1 > 0] = 1
    df1[df1 < 0] = -1
    df1[df1 == 0] = 0
    return df1.iloc[:,0]

def Max(sr,n):
    #比较取大函数
    df = sr.to_frame()
    df1 = df.copy()
    df1[df1 < n] = n
    return df1.iloc[:,0]

def Max2(sr1,sr2):
    #比较取小函数
    sr12 = sr1 - sr2
    df12 = sr12.to_frame()
    df12[df12 < 0] = 0
    sr12 = df12.iloc[:,0]
    sr12 = sr12 + sr2
    return sr12

def Min(sr,n):
    #比较取小函数
    df = sr.to_frame()
    df1 = df.copy()
    df1[df1 > n] = n
    return df1.iloc[:,0]

def Min2(sr1,sr2):
    #比较取小函数
    sr12 = sr1 - sr2
    df12 = sr12.to_frame()
    df12[df12 > 0] = 0
    sr12 = df12.iloc[:,0]
    sr12 = sr12 + sr2
    return sr12

def Sma(sr,n,m):
    #sma均值
    #df1 = df.ewm(alpha=m/n).mean()
    return sr.ewm(alpha=m/n, adjust=False).mean()

def Abs(sr):
    #求绝对值
    return sr.abs()

def Sequence(n):
    #生成 1~n 的等差序列
    #return np.arange(1,n+1)
    return pd.Series(np.arange(1,n+1).tolist())

def Regbeta(df,B,window):
    #回归求系数
    #temp=A.rolling(n).apply(lambda x:sp.stats.linregress(x,B)) 
    #result = sm.OLS(A,B).fit()
    df1 = df.copy()
    df1.iloc[0:window] = None
    for i in range(window,len(df)+1):
        result = df.iloc[i-window:i,:].apply(lambda x: sp.stats.linregress(x,B) ,axis=0)
        df1.iloc[i-1,:] = result.iloc[0,:]
    return df1

def Decaylinear(sr, window):  #将dataframe运算转成np数组运算
    weights = np.arange(1,window+1,1)
    y = weights / weights.sum()  #y是和为1的权重
    sr1 = sr.copy()
    
    for row in range(window - 1, sr.shape[0]):
        x = sr.iloc[row - window + 1: row + 1]
        sr1.iloc[row] = (x*y).sum()
    return sr1

def Lowday(sr,window):
    #计算sr前window期时间序列中最小值距离当前时点的间隔
    sr1 = sr.copy()
    sr1[0:window] = np.nan
    for i in range(window, len(sr)+1):
        sr1.iloc[i-1:i] = window - 1 - sr.iloc[i-window:i].argmin()
    return sr1

def Highday(sr,window):
    #计算sr前window期时间序列中最大值距离当前时点的间隔
    sr1 = sr.copy()
    sr1[0:window] = np.nan
    for i in range(window, len(sr)+1):
        sr1.iloc[i-1:i] = window - 1 - sr.iloc[i-window:i].argmax()
    return sr1

def Wma(sr,window):
    weights = 0.9*np.arange(window-1,0-1,-1)
    sr1 = sr.copy()
    for row in range(window-1, len(sr)):
        sr1.iloc[0:window-1] = np.nan
        x = sr.iloc[row-window+1:row+1]
        sr1.iloc[row] = (x*weights).sum()
    return sr1

def Count(part,window):
    #计算前n期满足条件condition的样本个数,此时输入的part为0、1变量
    part1 = pd.Series(np.zeros(part.shape))
    part1[0:window-1] = np.nan
    for i in range(window,len(part)+1): 
        part1.iloc[i-1:i] = part.iloc[i-20:i].value_counts().get(1)
    return part1

def Sumif(part,window):
    #对前n项条件求和，part为条件筛选后的数据
    part1 = pd.Series(np.zeros(part.shape))
    part1[0:window-1] = np.nan
    for i in range(window,len(part)+1): 
        part1.iloc[i-1:i] = part.iloc[i-window:i].sum()
    return part1

class Alphas:
    def __init__(self, stktrd):

        self.open = stktrd['open'] #开盘价
        self.high = stktrd['high'] #最高价
        self.low = stktrd['low']  #最低价
        self.close = stktrd['close']#收盘价
        self.close_prev = stktrd['close'].shift(1)#前一天收盘价
        self.volume = stktrd['volume']#交易量
        self.value = stktrd['volume']#stktrd['position']#公司总市值
        self.amount = stktrd['volume']#交易额
        self.returns = stktrd['close'].pct_change().dropna() #每日收益率
        self.vwap = stktrd['close']#stktrd['amount']/(stktrd['volume']+1)#交易均价
        self.benchmark_open = stktrd['open']#指数开盘价series
        self.benchmark_close = stktrd['close']#指数收盘价series
   
    def alpha_1(self): #平均1751个数据
        ##### (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))#### 
        return (-1 * Corr(Rank(Delta(Log(self.volume), 1)), Rank(((self.close - self.open) / self.open)), 6))
    
    def alpha_2(self): #1783
        ##### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
        return -1*Delta((((self.close-self.low)-(self.high-self.close))/(self.high-self.low)),1) 
    
    def alpha_3(self): 
        ##### SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6) ####
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = 0
        part[cond2] = self.close - Min2(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max2(self.high,Delay(self.close,1))
        return Sum(part, 6)
    
    def alpha_4(self):  
        #####((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        cond1 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) < Sum(self.close, 2)/2)
        cond2 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) > Sum(self.close, 2)/2)
        cond3 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) == Sum(self.close, 2)/2)
        cond4 = (self.volume/Mean(self.volume, 20) >= 1)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = -1
        part[cond2] = 1
        part[cond3][cond4] = 1
        part[cond3][~cond4] = -1
        
        return part
    
    def alpha_5(self): #1447
        ####(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))###
        return -1*Tsmax(Corr(Tsrank(self.volume, 5),Tsrank(self.high, 5),5), 3)
    
    def alpha_6(self): #1779
        ####(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)### 
        return -1*Rank(Sign(Delta(((self.open * 0.85) + (self.high * 0.15)), 4)))
    
    def alpha_7(self): #1782
        ####((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))###
        return ((Rank(Max((self.vwap - self.close), 3)) + Rank(Min((self.vwap - self.close), 3))) * Rank(Delta(self.volume, 3)))
    
    def alpha_8(self): #1779
        ####RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)###    
        return Rank(Delta(((((self.high + self.low) / 2) * 0.2) + (self.vwap * 0.8)), 4) * -1)
    
    def alpha_9(self): #1790
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)###  
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,7,2)
    
    def alpha_10(self):    
        ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))###
        cond = (self.returns < 0)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = Std(self.returns, 20)
        part[~cond] = self.close
        part = part**2
        
        return Rank(Max(part, 5))
    
    def alpha_11(self): #1782
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)###   
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,6)
    
    def alpha_12(self): #1779
        ####(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))###   
        return (Rank((self.open - (Sum(self.vwap, 10) / 10)))) * (-1 * (Rank(Abs((self.close - self.vwap)))))
    
    def alpha_13(self): #1790
        ####(((HIGH * LOW)^0.5) - VWAP)###
        return (((self.high * self.low)**0.5) - self.vwap)
    
    def alpha_14(self): #1776
        ####CLOSE-DELAY(CLOSE,5)###
        return self.close-Delay(self.close,5)
    
    def alpha_15(self): #1790
        ####OPEN/DELAY(CLOSE,1)-1###
        return self.open/Delay(self.close,1)-1
    
    def alpha_16(self): #1736   
        ####(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))###
        return (-1 * Tsmax(Rank(Corr(Rank(self.volume), Rank(self.vwap), 5)), 5))
        
    def alpha_17(self): #1776   
        ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)###
        return Rank((self.vwap - Max(self.vwap, 15)))**Delta(self.close, 5)
    
    def alpha_18(self): #1776   
        ####CLOSE/DELAY(CLOSE,5)###
        return self.close/Delay(self.close,5)  
    
    def alpha_19(self):  
        ####(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))###
        cond1 = (self.close < Delay(self.close,5))
        cond2 = (self.close == Delay(self.close,5))
        cond3 = (self.close > Delay(self.close,5))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = (self.close-Delay(self.close,5))/Delay(self.close,5)
        part[cond2] = 0
        part[cond3] = (self.close-Delay(self.close,5))/self.close
        
        return part
       
    def alpha_20(self): #1773      
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100###
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*100
    
    def alpha_21(self):  #reg？
        ####REGBETA(MEAN(CLOSE,6),SEQUENCE(6))###
        return 0
    
    def alpha_22(self): #1736  
        ####SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)###
        return Sma(((self.close-Mean(self.close,6))/Mean(self.close,6)-Delay((self.close-Mean(self.close,6))/Mean(self.close,6),3)),12,1)
     
    def alpha_23(self):  
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100###
        cond = (self.close > Delay(self.close,1))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = Std(self.close,20)
        part1[~cond] = 0
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = Std(self.close,20)
        part2[cond] = 0
        
        return 100*Sma(part1,20,1)/(Sma(part1,20,1) + Sma(part2,20,1))
        
    def alpha_24(self): #1776  
        ####SMA(CLOSE-DELAY(CLOSE,5),5,1)###
        return Sma(self.close-Delay(self.close,5),5,1)
    
    def alpha_25(self):  #886  数据量较少
        ####((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))###
        return ((-1 * Rank((Delta(self.close, 7) * (1 - Rank(Decaylinear((self.volume / Mean(self.volume,20)), 9)))))) * (1 + Rank(Sum(self.returns, 250))))
    
    def alpha_26(self):   #平均数据量914，获得的数据量较少 
        ####((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))###
        return ((((Sum(self.close, 7) / 7) - self.close)) + ((Corr(self.vwap, Delay(self.close, 5), 230))))
    
    def alpha_27(self):  
        ####WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)###
        A = (self.close-Delay(self.close,3))/Delay(self.close,3)*100+(self.close-Delay(self.close,6))/Delay(self.close,6)*100
        return Wma(A, 12)
    
    def alpha_28(self):   #1728 
        ####3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)###
        return 3*Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)-2*Sma(Sma((self.close-Tsmin(self.low,9))/(Max(self.high,9)-Tsmax(self.low,9))*100,3,1),3,1)
    
    def alpha_29(self):   #1773 
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME###
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*self.volume
    
    def alpha_30(self):  #reg？
        ####WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML， 60))^2,20)###
        return 0
    
    def alpha_31(self):   #1714
        ####(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100###
        return (self.close-Mean(self.close,12))/Mean(self.close,12)*100
    
    def alpha_32(self):   #1505
        ####(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))###
        return (-1 * Sum(Rank(Corr(Rank(self.high), Rank(self.volume), 3)), 3))
    
    def alpha_33(self):   #904  数据量较少
        ####((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))###
        return ((((-1 * Tsmin(self.low, 5)) + Delay(Tsmin(self.low, 5), 5)) * Rank(((Sum(self.returns, 240) - Sum(self.returns, 20)) / 220))) *Tsrank(self.volume, 5))
    
    def alpha_34(self):   #1714
        ####MEAN(CLOSE,12)/CLOSE###
        return Mean(self.close,12)/self.close
    
    def alpha_35(self):   #1790    (OPEN * 0.65) +(OPEN *0.35)有问题
        ####(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)###
        return (Min2(Rank(Decaylinear(Delta(self.open, 1), 15)), Rank(Decaylinear(Corr((self.volume), ((self.open * 0.65) +(self.open *0.35)), 17),7))) * -1)
     
    def alpha_36(self):   #1714
        ####RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP),6), 2))###
        return Rank(Sum(Corr(Rank(self.volume), Rank(self.vwap),6 ), 2))
    
    def alpha_37(self):   #1713
        ####(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))###
        return (-1 * Rank(((Sum(self.open, 5) * Sum(self.returns, 5)) - Delay((Sum(self.open, 5) * Sum(self.returns, 5)), 10))))
    
    def alpha_38(self):  
        ####(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        cond = ((Sum(self.high, 20) / 20) < self.high)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = -1 * Delta(self.high, 2)
        part[~cond] = 0
        
        return part
    
    def alpha_39(self):   #1666
        ####((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)###
        return ((Rank(Decaylinear(Delta((self.close), 2),8)) - Rank(Decaylinear(Corr(((self.vwap * 0.3) + (self.open * 0.7)),Sum(Mean(self.volume,180), 37), 14), 12))) * -1)
    
    def alpha_40(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100###
        cond = (self.close > Delay(self.close,1))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = self.volume
        part1[~cond] = 0
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = self.volume
        part2[cond] = 0
        
        return Sum(part1,26)/Sum(part2,26)*100
    
    def alpha_41(self):   #1782
        ####(RANK(MAX(DELTA((VWAP), 3), 5))* -1)###
        return (Rank(Max(Delta((self.vwap), 3), 5))* -1)
    
    def alpha_42(self):   #1399  数据量较少
        ####((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))###
        return ((-1 * Rank(Std(self.high, 10))) * Corr(self.high, self.volume, 10))
    
    def alpha_43(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = self.volume
        part[cond2] = -self.volume
        part[cond3] = 0
        
        return Sum(part,6)
    
    def alpha_44(self):   #1748
        ####(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))###
        return (Tsrank(Decaylinear(Corr(((self.low)), Mean(self.volume,10), 7), 6),4) + Tsrank(Decaylinear(Delta((self.vwap),3), 10), 15))
    
    def alpha_45(self):   #1070  数据量较少
        ####(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))###
        return (Rank(Delta((((self.close * 0.6) + (self.open *0.4))), 1)) * Rank(Corr(self.vwap, Mean(self.volume,150), 15)))
    
    def alpha_46(self):   #1630
        ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)###
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/(4*self.close)
    
    def alpha_47(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,9,1)
    
    def alpha_48(self):   #1657
        ####(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))###
        return (-1*((Rank(((Sign((self.close - Delay(self.close, 1))) + Sign((Delay(self.close, 1) - Delay(self.close, 2)))) + Sign((Delay(self.close, 2) - Delay(self.close, 3)))))) * Sum(self.volume, 5)) / Sum(self.volume, 20))
    
    def alpha_49(self):  
        ####SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) + SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        cond = ((self.high + self.low) > (Delay(self.high,1) + Delay(self.low,1)))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = 0
        part1[~cond] = Max2(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = 0
        part2[cond] = Max2(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))
    
    def alpha_50(self):  
        ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = 0
        part1[~cond] = Max2(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = 0
        part2[cond] = Max2(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return (Sum(part1, 12) - Sum(part2, 12)) / (Sum(part1, 12) + Sum(part2, 12)) 

    def alpha_51(self):  
        ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = 0
        part1[~cond] = Max2(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = 0
        part2[cond] = Max2(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))
    
    def alpha_52(self):   #1611
        ####SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100###
        return Sum(Max(self.high-Delay((self.high+self.low+self.close)/3,1),0),26)/Sum(Max(Delay((self.high+self.low+self.close)/3,1)-self.low, 0),26)*100
    
    def alpha_53(self):  
        ####COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100###
        cond = (self.close > Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 1 #把满足条件的记为1，之后统计1的个数
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[0:12] = np.nan
        for i in range(12,len(part1)+1): 
            part1.iloc[i-1:i] = part.iloc[i-12:i].value_counts().get(1)
        
        return part1
    
    def alpha_54(self):   #1729
        ####(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))###
        return (-1 * Rank(((Abs(self.close - self.open)).std() + (self.close - self.open)) + Corr(self.close, self.open,10)))
    
    def alpha_55(self):  #公式有问题
        ####SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2 + ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
        A = Abs(self.high - Delay(self.close, 1))
        B = Abs(self.low - Delay(self.close, 1))
        C = Abs(self.high - Delay(self.low, 1))
        cond1 = ((A > B) & (A > C))
        cond2 = ((B > C) & (B > A))
        cond3 = ((C >= A) & (C >= B))
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond1] = Abs(self.high - Delay(self.close, 1)) + Abs(self.low - Delay(self.close, 1))/2 + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part1[cond2] = Abs(self.low - Delay(self.close, 1)) + Abs(self.high - Delay(self.close, 1))/2 + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part1[cond3] = Abs(self.high - Delay(self.low, 1)) + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        
        return Sum(part0/part1,20)
    
    def alpha_56(self):  
        ####(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))###
        A = Rank((self.open - Tsmin(self.open, 12)))
        B = Rank((Rank(Corr(Sum(((self.high + self.low) / 2), 19),Sum(Mean(self.volume,40), 19), 13))**5))
        cond = (A < B)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 1
        #part[~cond] = 0
        return part
    
    def alpha_57(self):   #1736
        ####SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)###
        return Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)
    
    def alpha_58(self):  
        ####COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100###

        cond = (self.close > Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 1 #把满足条件的记为1，之后统计1的个数
        '''
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[0:19] = np.nan
        for i in range(20,len(part1)+1): 
            part1.iloc[i-1:i] = part.iloc[i-20:i].value_counts().get(1)
        return part1'''
        return Count(part,20)
        
    
    def alpha_59(self):  
        ####SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)###
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = 0
        part[cond2] = self.close - Min2(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max2(self.low,Delay(self.close,1))
        
        return Sum(part, 20)
    
    def alpha_60(self):   #1635
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)###
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,20)

    def alpha_61(self):   #1790
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)###
        return (Max2(Rank(Decaylinear(Delta(self.vwap, 1), 12)),Rank(Decaylinear(Rank(Corr((self.low),Mean(self.volume,80), 8)), 17))) * -1)
    
    def alpha_62(self):   #1479
        ####(-1 * CORR(HIGH, RANK(VOLUME), 5))###
        return (-1 * Corr(self.high, Rank(self.volume), 5))
    
    def alpha_63(self):   #1789
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),6,1)/Sma(Abs(self.close-Delay(self.close,1)),6,1)*100
    
    def alpha_64(self):   #1774
        ####(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)###
        return (Max2(Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 4), 4)),Rank(Decaylinear(Max(Corr(Rank(self.close), Rank(Mean(self.volume,60)), 4), 13), 14))) * -1)
    
    def alpha_65(self):   #1759
        ####MEAN(CLOSE,6)/CLOSE###
        return Mean(self.close,6)/self.close
    
    def alpha_66(self):   #1759
        ####(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100###
        return (self.close-Mean(self.close,6))/Mean(self.close,6)*100
    
    def alpha_67(self):   #1759
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),24,1)/Sma(Abs(self.close-Delay(self.close,1)),24,1)*100
    
    def alpha_68(self):   #1790
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)###
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,15,2)
    
    def alpha_69(self):  
        ####(SUM(DTM,20)>SUM(DBM,20)？ (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)： (SUM(DTM,20)=SUM(DBM,20)？0： (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))###
        ####DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        ####DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
        cond1 = (self.open <= Delay(self.open,1))
        cond2 = (self.open >= Delay(self.open,1))
        
        DTM = pd.Series(np.zeros(self.close.shape))
        #DTM[cond1] = 0
        DTM[~cond1] = Max2((self.high-self.open),(self.open-Delay(self.open,1)))
        
        DBM = pd.Series(np.zeros(self.close.shape))
        #DBM[cond2] = 0
        DBM[~cond2] = Max2((self.open-self.low),(self.open-Delay(self.open,1)))
        
        cond3 = (Sum(DTM,20) > Sum(DBM,20))
        cond4 = (Sum(DTM,20)== Sum(DBM,20))
        cond5 = (Sum(DTM,20) < Sum(DBM,20))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond3] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DTM,20)
        #part[cond4] = 0
        part[cond5] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DBM,20)
        return part
    
    def alpha_70(self):   #1759
        ####STD(AMOUNT,6)###
        return Std(self.amount,6)
    
    def alpha_71(self):   #1630
        ####(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100###
        return (self.close-Mean(self.close,24))/Mean(self.close,24)*100
    
    def alpha_72(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,15,1)
    
    def alpha_73(self):   #1729
        ####((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)###
        return ((Tsrank(Decaylinear(Decaylinear(Corr((self.close), self.volume, 10), 16), 4), 5) - Rank(Decaylinear(Corr(self.vwap, Mean(self.volume,30), 4),3))) * -1) 
    
    def alpha_74(self):   #1402
        ####(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))###
        return (Rank(Corr(Sum(((self.low * 0.35) + (self.vwap * 0.65)), 20), Sum(Mean(self.volume,40), 20), 7)) + Rank(Corr(Rank(self.vwap), Rank(self.volume), 6)))
    
    def alpha_75(self):  
        ####COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)###
        cond1 = ((self.close>self.open)&(self.benchmark_close<self.benchmark_open))
        cond2 = (self.benchmark_close<self.benchmark_open)
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond1] = 1
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[cond2] = 1
        
        return Count(part1,50)/Count(part2,50)
    
    def alpha_76(self):   #1650
        ####STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)###
        return Std(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)/Mean(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)
    
    def alpha_77(self):   #1797
        #### MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))###
        return  Min2(Rank(Decaylinear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)),Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 3), 6)))
       
    def alpha_78(self):   #1637
        ####((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))###
        return ((self.high+self.low+self.close)/3-Mean((self.high+self.low+self.close)/3,12))/(0.015*Mean(Abs(self.close-Mean((self.high+self.low+self.close)/3,12)),12))
    
    def alpha_79(self):   #1789
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100
    
    def alpha_80(self):   #1776
        ####(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100###
        return (self.volume-Delay(self.volume,5))/Delay(self.volume,5)*100
    
    def alpha_81(self):   #1797
        ####SMA(VOLUME,21,2)###
        return Sma(self.volume,21,2)
    
    def alpha_82(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,20,1)
    
    def alpha_83(self):   #1766
        ####(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))###
        return (-1 * Rank(Cov(Rank(self.high), Rank(self.volume), 5)))
    
    def alpha_84(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))  
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = self.volume
        part[cond2] = 0
        part[cond3] = -self.volume 
        return Sum(part, 20)
    
    def alpha_85(self):   #1657
        ####(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))###
        return (Tsrank((self.volume / Mean(self.volume,20)), 20) * Tsrank((-1 * Delta(self.close, 7)), 8))
    
    def alpha_86(self):  
        ####((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ?1 : ((-1 * 1) *(CLOSE - DELAY(CLOSE, 1)))))
        A = (((Delay(self.close, 20) - Delay(self.close, 10)) / 10) - ((Delay(self.close, 10) - self.close) / 10))
        cond1 = (A > 0.25)
        cond2 = (A < 0.0)
        cond3 = ((0 <= A) & (A <= 0.25))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1*(self.close - Delay(self.close, 1))
        return part

    def alpha_87(self):   #1741
        ####((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /(OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)###
        return ((Rank(Decaylinear(Delta(self.vwap, 4), 7)) + Tsrank(Decaylinear(((((self.low * 0.9) + (self.low * 0.1)) - self.vwap) /(self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
  
    def alpha_88(self):   #1745
        ####(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100###
        return (self.close-Delay(self.close,20))/Delay(self.close,20)*100
    
    def alpha_89(self):   #1797
        ####2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))###
        return 2*(Sma(self.close,13,2)-Sma(self.close,27,2)-Sma(Sma(self.close,13,2)-Sma(self.close,27,2),10,2))
    
    def alpha_90(self):   #1745
        ####(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)###
        return (Rank(Corr(Rank(self.vwap), Rank(self.volume), 5)) * -1)
    
    def alpha_91(self):   #1745
        ####((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)###
        return ((Rank((self.close - Max(self.close, 5)))*Rank(Corr((Mean(self.volume,40)), self.low, 5))) * -1)
    
    def alpha_92(self):   #1786
        ####(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)###
        return (Max2(Rank(Decaylinear(Delta(((self.close * 0.35) + (self.vwap *0.65)), 2), 3)),Tsrank(Decaylinear(Abs(Corr((Mean(self.volume,180)), self.close, 13)), 5), 15)) * -1)
    
    def alpha_93(self):  
        ####SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)###
        cond = (self.open >= Delay(self.open,1))
        part = pd.Series(np.zeros(self.close.shape))
        #part[cond] = 0
        part[~cond] = Max2((self.open-self.low),(self.open-Delay(self.open,1)))
        return Sum(part, 20)
    
    def alpha_94(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = self.volume
        part[cond2] = -1*self.volume
        #part[cond3] = 0
        return Sum(part, 30)
    
    def alpha_95(self):   #1657
        ####STD(AMOUNT,20)###
        return Std(self.amount,20)
    
    def alpha_96(self):   #1736
        ####SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)###
        return Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1),3,1)
    
    def alpha_97(self):   #1729
        ####STD(VOLUME,10)###
        return Std(self.volume,10)
    
    def alpha_98(self):  
        ####((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))###
        cond = (Delta(Sum(self.close,100)/100, 100)/Delay(self.close, 100) <= 0.05)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = -1 * (self.close - Tsmin(self.close, 100))
        part[~cond] = -1 * Delta(self.close, 3)
        return part
    
    def alpha_99(self):   #1766
        ####(-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))###
        return (-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))
    
    def alpha_100(self):   #1657
        ####Std(self.volume,20)###
        return Std(self.volume,20)
    
    def alpha_101(self):  
        ###((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
        rank1 = Rank(Corr(self.close, Sum(Mean(self.volume,30), 37), 15))
        rank2 = Rank(Corr(Rank(((self.high * 0.1) + (self.vwap * 0.9))),Rank(self.volume), 11))
        cond = (rank1<rank2)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 1
        #part[~cond] = 0
        return part
    
    def alpha_102(self):   #1790
        ####SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100###
        return Sma(Max(self.volume-Delay(self.volume,1),0),6,1)/Sma(Abs(self.volume-Delay(self.volume,1)),6,1)*100
    
    def alpha_103(self):  
        ####((20-LOWDAY(LOW,20))/20)*100###
        return ((20-Lowday(self.low,20))/20)*100
    
    def alpha_104(self):   #1657
        ####(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))###
        return (-1 * (Delta(Corr(self.high, self.volume, 5), 5) * Rank(Std(self.close, 20))))
    
    def alpha_105(self):   #1729
        ####(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))###
        return (-1 * Corr(Rank(self.open), Rank(self.volume), 10))
    
    def alpha_106(self):   #1745
        ####CLOSE-DELAY(CLOSE,20)###
        return self.close-Delay(self.close,20)
    
    def alpha_107(self):   #1790
        ####(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))###
        return (((-1 * Rank((self.open - Delay(self.high, 1)))) * Rank((self.open - Delay(self.close, 1)))) * Rank((self.open - Delay(self.low, 1))))
    
    def alpha_108(self):   #1178   
        ####((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)###
        return ((Rank((self.high - Min(self.high, 2)))**Rank(Corr((self.vwap), (Mean(self.volume,120)), 6))) * -1)
    
    def alpha_109(self):   #1797
        ####SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)###
        return Sma(self.high-self.low,10,2)/Sma(Sma(self.high-self.low,10,2),10,2)
    
    def alpha_110(self):   #1650
        ####SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100###
        return Sum(Max(self.high-Delay(self.close,1),0),20)/Sum(Max(Delay(self.close,1)-self.low,0),20)*100
      
    def alpha_111(self):   #1789
        ####SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)###
        return Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),11,2)-Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),4,2)
    
    def alpha_112(self):  
        ####(SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12) + SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100     
        cond = (self.close-Delay(self.close,1) > 0)
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = self.close-Delay(self.close,1)
        #part1[~cond] = 0
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = Abs(self.close-Delay(self.close,1))
        #part2[cond] = 0
        return (Sum(part1,12) - Sum(part2,12))/(Sum(part1,12) + Sum(part2,12))*100
    
    def alpha_113(self):   #1587
        ####(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))###
        return (-1 * ((Rank((Sum(Delay(self.close, 5), 20) / 20)) * Corr(self.close, self.volume, 2)) * Rank(Corr(Sum(self.close, 5),Sum(self.close, 20), 2))))
    
    def alpha_114(self):   #1751
        ####((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))###
        return ((Rank(Delay(((self.high - self.low) / (Sum(self.close, 5) / 5)), 2)) * Rank(Rank(self.volume))) / (((self.high - self.low) /(Sum(self.close, 5) / 5)) / (self.vwap - self.close)))
    
    def alpha_115(self):   #1527
        ####(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))###
        return (Rank(Corr(((self.high * 0.9) + (self.close * 0.1)), Mean(self.volume,30), 10))**Rank(Corr(Tsrank(((self.high + self.low) /2), 4), Tsrank(self.volume, 10), 7)))
    
    def alpha_116(self):  
        ####REGBETA(CLOSE,SEQUENCE,20)###
        return 0
    
    def alpha_117(self):   #1786
        ####((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))###
        return ((Tsrank(self.volume, 32) * (1 - Tsrank(((self.close + self.high) - self.low), 16))) * (1 - Tsrank(self.returns, 32)))
    
    def alpha_118(self):   #1657
        ####SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100###
        return Sum(self.high-self.open,20)/Sum(self.open-self.low,20)*100
    
    def alpha_119(self):   #1626
        ####(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))###
        return (Rank(Decaylinear(Corr(self.vwap, Sum(Mean(self.volume,5), 26), 5), 7)) - Rank(Decaylinear(Tsrank(Min(Corr(Rank(self.open), Rank(Mean(self.volume,15)), 21), 9), 7), 8)))
    
    def alpha_120(self):   #1797
        ####(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))###
        return (Rank((self.vwap - self.close)) / Rank((self.vwap + self.close)))
    
    def alpha_121(self):   #972   数据量较少
        ####((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)###
        return ((Rank((self.vwap - Min(self.vwap, 12)))**Tsrank(Corr(Tsrank(self.vwap, 20), Tsrank(Mean(self.volume,60), 2), 18), 3)) *-1)
    
    def alpha_122(self):   #1790
        ####(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)###
        return (Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)-Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1))/Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1)
    
    def alpha_123(self):  
        ####((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1)###
        A = Rank(Corr(Sum(((self.high + self.low) / 2), 20), Sum(Mean(self.volume,60), 20), 9))
        B = Rank(Corr(self.low, self.volume,6))
        cond = (A < B)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = -1
        #part[~cond] = 0
        return part
    
    def alpha_124(self):   #1592
        ####(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)###
        return (self.close - self.vwap) / Decaylinear(Rank(Tsmax(self.close, 30)),2)
     
    def alpha_125(self):   #1678
        ####(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))###
        return (Rank(Decaylinear(Corr((self.vwap), Mean(self.volume,80),17), 20)) / Rank(Decaylinear(Delta(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)))
    
    def alpha_126(self):   #1797
        ####(CLOSE+HIGH+LOW)/3###
        return (self.close+self.high+self.low)/3
    
    def alpha_127(self):  #公式有问题，我们假设mean周期为12
        ####(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2)###
        return (Mean((100*(self.close-Max(self.close,12))/(Max(self.close,12)))**2,12))**(1/2)
    
    def alpha_128(self):  
        ####100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
        A = (self.high+self.low+self.close)/3
        cond = (A > Delay(A,1))        
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = A*self.volume
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = A*self.volume
        return 100-(100/(1+Sum(part1,14)/Sum(part2,14)))

    def alpha_129(self):  
        ####SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)###
        cond = ((self.close-Delay(self.close,1)) < 0)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = Abs(self.close-Delay(self.close,1))
        #part[~cond] = 0
        return Sum(part, 12)
    
    def alpha_130(self):   #1657
        ####(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))###
        return (Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 9), 10)) / Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 7),3)))
    
    def alpha_131(self):   #1030   数据量较少
        ####(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))###
        return (Rank(Delta(self.vwap, 1))**Tsrank(Corr(self.close,Mean(self.volume,50), 18), 18))
       
    def alpha_132(self):   #1657
        ####MEAN(AMOUNT,20)###
        return Mean(self.amount,20)
    
    def alpha_133(self):  
        ####((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100###
        return ((20-Highday(self.high,20))/20)*100-((20-Lowday(self.low,20))/20)*100
    
    def alpha_134(self):   #1760
        ####(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME###
        return (self.close-Delay(self.close,12))/Delay(self.close,12)*self.volume
    
    def alpha_135(self):   #1744
        ####SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)###
        return Sma(Delay(self.close/Delay(self.close,20),1),20,1)
    
    def alpha_136(self):   #1729
        ####((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))###
        return ((-1 * Rank(Delta(self.returns, 3))) * Corr(self.open, self.volume, 10))
    
    def alpha_137(self):  
        ####16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
        A = Abs(self.high- Delay(self.close,1))
        B = Abs(self.low - Delay(self.close,1))
        C = Abs(self.high- Delay(self.low,1))
        D = Abs(Delay(self.close,1)-Delay(self.open,1))          
        cond1 = ((A>B) & (A>C))
        cond2 = ((B>C) & (B>A))
        cond3 = ((C>=A) & (C>=B))       
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond1] = A + B/2 + D/4
        part1[cond2] = B + A/2 + D/4
        part1[cond3] = C + D/4     
        return part0/part1*Max2(A,B)

    def alpha_138(self):   #1448
        ####((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)###
        return ((Rank(Decaylinear(Delta((((self.low * 0.7) + (self.vwap *0.3))), 3), 20)) - Tsrank(Decaylinear(Tsrank(Corr(Tsrank(self.low, 8), Tsrank(Mean(self.volume,60), 17), 5), 19), 16), 7)) * -1)
    
    def alpha_139(self):   #1729
        ####(-1 * CORR(OPEN, VOLUME, 10))###
        return (-1 * Corr(self.open, self.volume, 10))
    
    def alpha_140(self):   #1797
        ####MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))###
        return Min2(Rank(Decaylinear(((Rank(self.open) + Rank(self.low)) - (Rank(self.high) + Rank(self.close))), 8)), Tsrank(Decaylinear(Corr(Tsrank(self.close, 8), Tsrank(Mean(self.volume,60), 20), 8), 7), 3))
    
    def alpha_141(self):   #1637
        ####(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)###
        return (Rank(Corr(Rank(self.high), Rank(Mean(self.volume,15)), 9))* -1)
    
    def alpha_142(self):   #1657
        ####(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))###
        return (((-1 * Rank(Tsrank(self.close, 10))) * Rank(Delta(Delta(self.close, 1), 1))) * Rank(Tsrank((self.volume/Mean(self.volume,20)), 5)))
    
    def alpha_143(self):  
        ####CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF###

        return 0
    
    def alpha_144(self):  
        ####SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)###
        cond = (self.close<Delay(self.close,1))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = Abs(self.close/Delay(self.close,1)-1)/self.amount
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[cond] = 1
        return Sumif(part1,20)/Count(part2,20)
    
    def alpha_145(self):   #1617
        ####(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100###
        return (Mean(self.volume,9)-Mean(self.volume,26))/Mean(self.volume,12)*100
    
    def alpha_146(self):   #1650  公式有问题
        ####MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,61,2)###
        return Mean((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2),20)*((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2))/Sma(((self.close-Delay(self.close,1))/Delay(self.close,1)-((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2)))**2,61,2)

    def alpha_147(self):  
        ####REGBETA(MEAN(CLOSE,12),SEQUENCE(12))###
        return 0
    
    def alpha_148(self):  
        ####((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)###
        cond = (Rank(Corr((self.open), Sum(Mean(self.volume,60), 9), 6)) < Rank((self.open - Tsmin(self.open, 14))))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = -1
        #part[~cond] = 0
        return part
    
    def alpha_149(self):  
        ####REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
        return 0
    
    def alpha_150(self):   #1797
        ####(CLOSE+HIGH+LOW)/3*VOLUME###
        return (self.close+self.high+self.low)/3*self.volume
    
    def alpha_151(self):   #1745
        ####SMA(CLOSE-DELAY(CLOSE,20),20,1)###
        return Sma(self.close-Delay(self.close,20),20,1)
    
    def alpha_152(self):   #1559
        ####SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)###
        return Sma(Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),12)-Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),26),9,1)
    
    def alpha_153(self):   #1630
        ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4###
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/4
    
    def alpha_154(self):  
        ####(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))###
        cond = (((self.vwap - Min(self.vwap, 16))) < (Corr(self.vwap, Mean(self.volume,180), 18)))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 1
        return part
    
    def alpha_155(self):   #1797
        ####SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)###
        return Sma(self.volume,13,2)-Sma(self.volume,27,2)-Sma(Sma(self.volume,13,2)-Sma(self.volume,27,2),10,2)
    
    def alpha_156(self):   #1776
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)###
        return (Max2(Rank(Decaylinear(Delta(self.vwap, 5), 3)), Rank(Decaylinear(((Delta(((self.open * 0.15) + (self.low *0.85)),2) / ((self.open * 0.15) + (self.low * 0.85))) * -1), 3))) * -1)
    
    def alpha_157(self):   #1764
        ####(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))###
        return (Min(Prod(Rank(Rank(Log(Sum(Tsmin(Rank(Rank((-1 * Rank(Delta((self.close - 1), 5))))), 2), 1)))), 1), 5) + Tsrank(Delay((-1 * self.returns), 6), 5))
    
    def alpha_158(self):   #1797
        ####((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE###
        return ((self.high-Sma(self.close,15,2))-(self.low-Sma(self.close,15,2)))/self.close
    
    def alpha_159(self):   #1630
        ####((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)###
        return ((self.close-Sum(Min2(self.low,Delay(self.close,1)),6))/Sum(Max2(self.high,Delay(self.close,1))-Min2(self.low,Delay(self.close,1)),6)*12*24+(self.close-Sum(Min2(self.low,Delay(self.close,1)),12))/Sum(Max2(self.high,Delay(self.close,1))-Min2(self.low,Delay(self.close,1)),12)*6*24+(self.close-Sum(Min2(self.low,Delay(self.close,1)),24))/Sum(Max2(self.high,Delay(self.close,1))-Min2(self.low,Delay(self.close,1)),24)*6*24)*100/(6*12+6*24+12*24)
    
    def alpha_160(self):  
        ####SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
        cond = (self.close<=Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = Std(self.close,20)
        #part[~cond] = 0
        return Sma(part, 20, 1)
    
    def alpha_161(self):   #1714
        ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)###
        return Mean(Max2(Max2((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),12)
    
    def alpha_162(self):   #1789
        ####(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))###
        return (Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100-Min(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))/(Sma(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12,1)-Min(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))
    
    def alpha_163(self):   #1657
        ####RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))###
        return Rank(((((-1 * self.returns) * Mean(self.volume,20)) * self.vwap) * (self.high - self.close)))
    
    def alpha_164(self):  
        ####SMA(( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) - MIN( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) ,12) )/(HIGH-LOW)*100,13,2)###
        cond = (self.close>Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 1/(self.close-Delay(self.close,1))
        part[~cond] = 1
        return Sma((part - Min(part,12))/(self.high-self.low)*100, 13, 2)
    
    def alpha_165(self):  
        ####MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)###
        
        return 0
    
    def alpha_166(self):  #公式有问题
        
        return 0

    def alpha_167(self):  
        ####SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)###
        cond = (self.close > Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = self.close-Delay(self.close,1)
        #part[~cond] = 0
        return Sum(part,12)
    
    def alpha_168(self):   #1657
        ####(-1*VOLUME/MEAN(VOLUME,20))###
        return (-1*self.volume/Mean(self.volume,20))
    
    def alpha_169(self):   #1610
        ####SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)###
        return Sma(Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),12)-Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),26),10,1)
    
    def alpha_170(self):   #1657
        ####((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))###
        return ((((Rank((1 / self.close)) * self.volume) / Mean(self.volume,20)) * ((self.high * Rank((self.high - self.close))) / (Sum(self.high, 5) /5))) - Rank((self.vwap - Delay(self.vwap, 5))))
   
    def alpha_171(self):   #1789
        ####((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))###
        return ((-1 * ((self.low - self.close) * (self.open**5))) / ((self.close - self.high) * (self.close**5)))
    
    def alpha_172(self):  
        ####MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        TR = Max2(Max2(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD)) 
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond1] = LD
        #part1[~cond1] = 0
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[cond2] = HD
        #part2[~cond2] = 0
        return Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)
    
    def alpha_173(self):   #1797
        ####3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)###
        return 3*Sma(self.close,13,2)-2*Sma(Sma(self.close,13,2),13,2)+Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)
    
    def alpha_174(self):  
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
        cond = (self.close>Delay(self.close,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = Std(self.close,20)
        #part[~cond] = 0
        return Sma(part,20,1)
    
    def alpha_175(self):
        ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)###
        return Mean(Max2(Max2((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),6)
    
    def alpha_176(self):   #1678
        ####CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)###
        return Corr(Rank(((self.close - Tsmin(self.low, 12)) / (Tsmax(self.high, 12) - Tsmin(self.low,12)))), Rank(self.volume), 6)
    
    def alpha_177(self):  
        ####((20-HIGHDAY(HIGH,20))/20)*100###
        return ((20-Highday(self.high,20))/20)*100
    
    def alpha_178(self):
        ####(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME###
        return (self.close-Delay(self.close,1))/Delay(self.close,1)*self.volume
    
    def alpha_179(self):
        ####(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))###
        return (Rank(Corr(self.vwap, self.volume, 4)) *Rank(Corr(Rank(self.low), Rank(Mean(self.volume,50)), 12)))
    
    def alpha_180(self):  #指标有问题
        ####((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 *VOLUME)))
        cond = (Mean(self.volume,20) < self.volume)
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = (-1 * Tsrank(Abs(Delta(self.close, 7)), 60)) * Sign(Delta(self.close, 7)) 
        part[~cond] = -1 * self.volume
        return part
    
    def alpha_181(self):   #公式有问题，假设后面的sum周期为20
        ####SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)###
        return Sum(((self.close/Delay(self.close,1)-1)-Mean((self.close/Delay(self.close,1)-1),20))-(self.benchmark_close-Mean(self.benchmark_close,20))**2,20)/Sum(((self.benchmark_close-Mean(self.benchmark_close,20))**3),20)
    
    def alpha_182(self):  
        ####COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20###
        cond1 = ((self.close>self.open) & (self.benchmark_close>self.benchmark_open))
        cond2 = ((self.close<self.open) & (self.benchmark_close<self.benchmark_open))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond1] = 1
        part[cond2] = 1
        return Count(part,20)/20
    
    def alpha_183(self):  
        ####MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)###
        return 0
    
    def alpha_184(self):
        ####(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))###
        return (Rank(Corr(Delay((self.open - self.close), 1), self.close, 200)) + Rank((self.open - self.close)))
    
    def alpha_185(self):
        ####RANK((-1 * ((1 - (OPEN / CLOSE))^2)))###
        return Rank((-1 * ((1 - (self.open / self.close))**2)))
    
    def alpha_186(self):  
        ####(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
        TR = Max2(Max2(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD)) 
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond1] = LD
        #part1[~cond1] = 0
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[cond2] = HD
        #part2[~cond2] = 0
        return (Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)+Delay(Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6),6))/2
    
    def alpha_187(self):  
        ####SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)###
        cond = (self.open<=Delay(self.open,1))
        part = pd.Series(np.zeros(self.close.shape))
        part[cond] = 0
        part[~cond] = Max2((self.high-self.open),(self.open-Delay(self.open,1)))
        return Sum(part,20) 
    
    def alpha_188(self):   #1797
        ####((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100###
        return ((self.high-self.low-Sma(self.high-self.low,11,2))/Sma(self.high-self.low,11,2))*100
    
    def alpha_189(self):   #1721
        ####MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)###
        return Mean(Abs(self.close-Mean(self.close,6)),6)
    
    def alpha_190(self):  #公式有大问题,
        ####LOG((COUNT( CLOSE/DELAY(CLOSE,1)>((CLOSE/DELAY(CLOSE,19))^(1/20)-1) ,20)-1)*(SUMIF((CLOSE/DELAY(CLOSE,1)-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE,1)-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
        '''
        cond = ((self.close/Delay(self.close,1)) > ((self.close/Delay(self.close,19))**(1/20)-1))
        part1 = pd.Series(np.zeros(self.close.shape))
        part1[cond] = 1 #COUNT
        part2 = pd.Series(np.zeros(self.close.shape))
        part2[~cond] = (self.close/Delay(self.close,1)-((self.close/Delay(self.close,19))**(1/20)-1))**2#SUMIF
        part3 = pd.Series(np.zeros(self.close.shape))
        part3[~cond] = 1 #COUNT
        part4 = pd.Series(np.zeros(self.close.shape))
        part4[cond] = (self.close/Delay(self.close,1)-((self.close/Delay(self.close,19))**(1/20)-1))**2#SUMIF
        return Log((Count(part1,20))*Sumif(part2,20)/(Count(part3,20)*Sumif(part4,20)))'''
        return 0
    
    def alpha_191(self):   #1721
        ####((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)###
        return ((Corr(Mean(self.volume,20), self.low, 5) + ((self.high + self.low) / 2)) - self.close)

def GET_alpha(df):
        stock=Alphas(df)
        df1 = df.copy()
        df2 = df[['date']].copy()
        df3 = df[['date']].copy()
        df4 = df[['date']].copy()
        #print('Alpha 191 Calculation Starts')
        df1['alpha_191_1']=stock.alpha_1() 
        df1['alpha_191_2']=stock.alpha_2()
        #df1['alpha_191_3']=stock.alpha_3()
        #df1['alpha_191_4']=stock.alpha_4()
        df1['alpha_191_5']=stock.alpha_5()
        df1['alpha_191_6']=stock.alpha_6()
        df1['alpha_191_7']=stock.alpha_7()
        df1['alpha_191_8']=stock.alpha_8()
        df1['alpha_191_9']=stock.alpha_9()
        #df1['alpha_191_10']=stock.alpha_10()
        df1['alpha_191_11']=stock.alpha_11() 
        df1['alpha_191_12']=stock.alpha_12()
        df1['alpha_191_13']=stock.alpha_13()
        df1['alpha_191_14']=stock.alpha_14()
        df1['alpha_191_15']=stock.alpha_15()
        df1['alpha_191_16']=stock.alpha_16()
        df1['alpha_191_17']=stock.alpha_17()
        df1['alpha_191_18']=stock.alpha_18()
        #df1['alpha_191_19']=stock.alpha_19()
        #print('Alpha 191 Calculation 10% Done')
        df1['alpha_191_20']=stock.alpha_20()
        #df1['alpha_191_21']=stock.alpha_21() 
        df1['alpha_191_22']=stock.alpha_22()
        #df1['alpha_191_23']=stock.alpha_23()
        df1['alpha_191_24']=stock.alpha_24()
        df1['alpha_191_25']=stock.alpha_25()
        df1['alpha_191_26']=stock.alpha_26()
        #df1['alpha_191_27']=stock.alpha_27()
        df1['alpha_191_28']=stock.alpha_28()
        df1['alpha_191_29']=stock.alpha_29()
        #df1['alpha_191_30']=stock.alpha_30()
        df1['alpha_191_31']=stock.alpha_31() 
        df1['alpha_191_32']=stock.alpha_32()
        df1['alpha_191_33']=stock.alpha_33()
        df1['alpha_191_34']=stock.alpha_34()
        df1['alpha_191_35']=stock.alpha_35()
        df1['alpha_191_36']=stock.alpha_36()
        df1['alpha_191_37']=stock.alpha_37()
        #df1['alpha_191_38']=stock.alpha_38()
        #print('Alpha 191 Calculation 20% Done')
        df1['alpha_191_39']=stock.alpha_39()
        #df1['alpha_191_40']=stock.alpha_40()
        df1['alpha_191_41']=stock.alpha_41() 
        df1['alpha_191_42']=stock.alpha_42()
        #df1['alpha_191_43']=stock.alpha_43()
        df1['alpha_191_44']=stock.alpha_44()
        df1['alpha_191_45']=stock.alpha_45()
        df1['alpha_191_46']=stock.alpha_46()
        df1['alpha_191_47']=stock.alpha_47()
        df1['alpha_191_48']=stock.alpha_48()
        #df1['alpha_191_49']=stock.alpha_49()
        #df1['alpha_191_50']=stock.alpha_50()
        #df2['alpha_191_51']=stock.alpha_51() 
        df2['alpha_191_52']=stock.alpha_52()
        #df2['alpha_191_53']=stock.alpha_53()
        df2['alpha_191_54']=stock.alpha_54()
        #df2['alpha_191_55']=stock.alpha_55()
        #df2['alpha_191_56']=stock.alpha_56()
        df2['alpha_191_57']=stock.alpha_57()
        #print('Alpha 191 Calculation 30% Done')
        #df2['alpha_191_58']=stock.alpha_58()
        #df2['alpha_191_59']=stock.alpha_59()
        df2['alpha_191_60']=stock.alpha_60()
        df2['alpha_191_61']=stock.alpha_61() 
        df2['alpha_191_62']=stock.alpha_62()
        df2['alpha_191_63']=stock.alpha_63()
        df2['alpha_191_64']=stock.alpha_64()
        df2['alpha_191_65']=stock.alpha_65()
        df2['alpha_191_66']=stock.alpha_66()
        df2['alpha_191_67']=stock.alpha_67()
        df2['alpha_191_68']=stock.alpha_68()
        #df2['alpha_191_69']=stock.alpha_69()
        df2['alpha_191_71']=stock.alpha_71() 
        df2['alpha_191_72']=stock.alpha_72()
        df2['alpha_191_73']=stock.alpha_73()
        df2['alpha_191_74']=stock.alpha_74()
        #df2['alpha_191_75']=stock.alpha_75()
        df2['alpha_191_76']=stock.alpha_76()
        #print('Alpha 191 Calculation 40% Done')
        df2['alpha_191_77']=stock.alpha_77()
        df2['alpha_191_78']=stock.alpha_78()
        df2['alpha_191_79']=stock.alpha_79()
        df2['alpha_191_80']=stock.alpha_80()
        df2['alpha_191_81']=stock.alpha_81() 
        df2['alpha_191_82']=stock.alpha_82()
        df2['alpha_191_83']=stock.alpha_83()
        #df2['alpha_191_84']=stock.alpha_84()
        df2['alpha_191_85']=stock.alpha_85()
        #df2['alpha_191_86']=stock.alpha_86()
        df2['alpha_191_87']=stock.alpha_87()
        df2['alpha_191_88']=stock.alpha_88()
        df2['alpha_191_89']=stock.alpha_89()
        df2['alpha_191_90']=stock.alpha_90()
        df2['alpha_191_91']=stock.alpha_91() 
        df2['alpha_191_92']=stock.alpha_92()
        #df2['alpha_191_93']=stock.alpha_93()
        #df2['alpha_191_94']=stock.alpha_94()
        df2['alpha_191_95']=stock.alpha_95()
        #print('Alpha 191 Calculation 50% Done')
        df2['alpha_191_96']=stock.alpha_96()
        df2['alpha_191_97']=stock.alpha_97()
        #df2['alpha_191_98']=stock.alpha_98()
        df2['alpha_191_99']=stock.alpha_99()
        df2['alpha_191_100']=stock.alpha_100()
        #df3['alpha_191_101']=stock.alpha_101() 
        df3['alpha_191_102']=stock.alpha_102()
        #df3['alpha_191_103']=stock.alpha_103()
        df3['alpha_191_104']=stock.alpha_104()
        df3['alpha_191_105']=stock.alpha_105()
        df3['alpha_191_106']=stock.alpha_106()
        df3['alpha_191_107']=stock.alpha_107()
        df3['alpha_191_108']=stock.alpha_108()
        df3['alpha_191_109']=stock.alpha_109()
        df3['alpha_191_110']=stock.alpha_110()
        df3['alpha_191_111']=stock.alpha_111() 
        #df3['alpha_191_112']=stock.alpha_112()
        df3['alpha_191_113']=stock.alpha_113()
        df3['alpha_191_114']=stock.alpha_114()
        #print('Alpha 191 Calculation 60% Done')
        df3['alpha_191_115']=stock.alpha_115()
        #df3['alpha_191_116']=stock.alpha_116()
        df3['alpha_191_117']=stock.alpha_117()
        df3['alpha_191_118']=stock.alpha_118()
        df3['alpha_191_119']=stock.alpha_119()
        df3['alpha_191_120']=stock.alpha_120()
        df3['alpha_191_121']=stock.alpha_121() 
        df3['alpha_191_122']=stock.alpha_122()
        #df3['alpha_191_123']=stock.alpha_123()
        df3['alpha_191_124']=stock.alpha_124()
        df3['alpha_191_125']=stock.alpha_125()
        df3['alpha_191_126']=stock.alpha_126()
        #df3['alpha_191_127']=stock.alpha_127()
        #df3['alpha_191_128']=stock.alpha_128()
        #df3['alpha_191_129']=stock.alpha_129()
        df3['alpha_191_130']=stock.alpha_130()
        df3['alpha_191_131']=stock.alpha_131() 
        #df3['alpha_191_133']=stock.alpha_133()
        df3['alpha_191_134']=stock.alpha_134()
        #print('Alpha 191 Calculation 70% Done')
        df3['alpha_191_135']=stock.alpha_135()
        df3['alpha_191_136']=stock.alpha_136()
        #df3['alpha_191_137']=stock.alpha_137()
        df3['alpha_191_138']=stock.alpha_138()
        df3['alpha_191_139']=stock.alpha_139()
        df3['alpha_191_140']=stock.alpha_140()
        df3['alpha_191_141']=stock.alpha_141() 
        df3['alpha_191_142']=stock.alpha_142()
        #df3['alpha_191_143']=stock.alpha_143()
        #df3['alpha_191_144']=stock.alpha_144()
        df3['alpha_191_145']=stock.alpha_145()
        df3['alpha_191_146']=stock.alpha_146()
        #df3['alpha_191_147']=stock.alpha_147()
        #df3['alpha_191_148']=stock.alpha_148()
        #df3['alpha_191_149']=stock.alpha_149()
        df3['alpha_191_150']=stock.alpha_150()
        df4['alpha_191_151']=stock.alpha_151() 
        df4['alpha_191_152']=stock.alpha_152()
        df4['alpha_191_153']=stock.alpha_153()
        #print('Alpha 191 Calculation 80% Done')
        #df4['alpha_191_154']=stock.alpha_154()
        df4['alpha_191_155']=stock.alpha_155()
        df4['alpha_191_156']=stock.alpha_156()
        df4['alpha_191_157']=stock.alpha_157()
        df4['alpha_191_158']=stock.alpha_158()
        df4['alpha_191_159']=stock.alpha_159()
        #df4['alpha_191_160']=stock.alpha_160()
        df4['alpha_191_161']=stock.alpha_161() 
        df4['alpha_191_162']=stock.alpha_162()
        df4['alpha_191_163']=stock.alpha_163()
        #df4['alpha_191_164']=stock.alpha_164()
        #df4['alpha_191_165']=stock.alpha_165()
        #df4['alpha_191_166']=stock.alpha_166()
        #df4['alpha_191_167']=stock.alpha_167()
        df4['alpha_191_168']=stock.alpha_168()
        df4['alpha_191_169']=stock.alpha_169()
        df4['alpha_191_170']=stock.alpha_170()
        df4['alpha_191_171']=stock.alpha_171() 
        #df4['alpha_191_172']=stock.alpha_172()
        #print('Alpha 191 Calculation 90% Done')
        df4['alpha_191_173']=stock.alpha_173()
        #df4['alpha_191_174']=stock.alpha_174()
        df4['alpha_191_175']=stock.alpha_175()
        df4['alpha_191_176']=stock.alpha_176()
        #df4['alpha_191_177']=stock.alpha_177()
        df4['alpha_191_178']=stock.alpha_178()
        df4['alpha_191_179']=stock.alpha_179()
        #df4['alpha_191_180']=stock.alpha_180()
        df4['alpha_191_181']=stock.alpha_181() 
        #df4['alpha_191_182']=stock.alpha_182()
        #df4['alpha_191_183']=stock.alpha_183()
        df4['alpha_191_184']=stock.alpha_184()
        df4['alpha_191_185']=stock.alpha_185()
        #df4['alpha_191_186']=stock.alpha_186()
        #df4['alpha_191_187']=stock.alpha_187()
        df4['alpha_191_188']=stock.alpha_188()
        df4['alpha_191_189']=stock.alpha_189()
        #df4['alpha_191_190']=stock.alpha_190()
        df4['alpha_191_191']=stock.alpha_191() 
        df2['alpha_191_70']=stock.alpha_70()
        df3['alpha_191_132']=stock.alpha_132()
        #print('Alpha 191 Calculation All Done')
        df12 = pd.merge(df1, df2, on = ['date'])
        df34 = pd.merge(df3, df4, on = ['date'])
        df1234 = pd.merge(df12, df34, on = ['date'])
        
        return df1234