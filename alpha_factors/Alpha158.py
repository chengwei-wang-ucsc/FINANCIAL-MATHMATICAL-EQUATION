import pandas as pd
import numpy as np
import talib

class Alphas158(object):
    def __init__(self, df):
        self.df = df

    def kmid(self):
        return (self.df.close-self.df.open)/self.df.open

    def klen(self):
        return (self.df.high-self.df.low)/self.df.open

    def kmid2(self):
        return (self.df.close-self.df.open)/(self.df.high-self.df.low+1e-12)
    
    def kup(self):
        res = []
        for i in range(len(self.df)):
            tmp_max = max(self.df.open.iloc[i], self.df.close.iloc[i])
            tmp_res = (self.df.high.iloc[i]-tmp_max)/self.df.open.iloc[i]
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kup2(self):
        res = []
        for i in range(len(self.df)):
            tmp_max = max(self.df.open.iloc[i], self.df.close.iloc[i])
            tmp_res = (self.df.high.iloc[i]-tmp_max)/(self.df.high.iloc[i]-self.df.low.iloc[i]+1e-12)
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow(self):
        res = []
        for i in range(len(self.df)):
            tmp_min = min(self.df.open.iloc[i], self.df.close.iloc[i])
            tmp_res = (tmp_min-self.df.low.iloc[i])/self.df.open.iloc[i]
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow2(self):
        res = []
        for i in range(len(self.df)):
            tmp_min = min(self.df.open.iloc[i], self.df.close.iloc[i])
            tmp_res = (tmp_min-self.df.low.iloc[i])/(self.df.high.iloc[i]-self.df.low.iloc[i]+1e-12)
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def ksft(self):
        return (2*self.df.close-self.df.high-self.df.low)/self.df.open

    def ksft2(self):
        res = (2*self.df.close-self.df.high-self.df.low)/(self.df.high-self.df.low+1e-12)
        res = res.to_numpy()
        return res

    def kmax_3(self):
        window = 3
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_5(self):
        window = 5
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_8(self):
        window = 8
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_13(self):
        window = 13
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_21(self):
        window = 21
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_34(self):
        window = 34
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_55(self):
        window = 55
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def kmax_88(self):
        window = 88
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_max = self.df.high.iloc[i:i+window].max()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_max/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_3(self):
        window = 3
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_5(self):
        window = 5
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_8(self):
        window = 8
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_13(self):
        window = 13
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_34(self):
        window = 34
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_55(self):
        window = 55
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def klow_88(self):
        window = 88
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_min = self.df.low.iloc[i:i+window].min()
            tmp_close = self.df.close.iloc[i+window]
            tmp_res = tmp_min/tmp_close
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_3(self):
        window = 3
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_5(self):
        window = 5
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_9(self):
        window = 9
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_13(self):
        window = 13
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_34(self):
        window = 34
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_55(self):
        window = 55
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def rsv_88(self):
        window = 88
        res = [0]*window
        for i in range(len(self.df.close)-window):
            tmp_close = self.df.close.iloc[i+window]
            n_min_low = self.df.low.iloc[i:i+window].min()
            n_max_high = self.df.high.iloc[i:i+window].max()
            tmp_res = (tmp_close-n_min_low)/(n_max_high-n_min_low)*100
            res.append(tmp_res)
        res = np.asarray(res)
        return res

    def corr_c_log_v_3(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 3
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_c_log_v_5(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 5
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_c_log_v_8(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 8
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_c_log_v_13(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 13
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_c_log_v_34(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 34
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_c_log_v_55(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 55
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_c_log_v_88(self):
        # The correlation between absolute close price and log scaled trading volume
        window = 88
        res = np.nan_to_num(talib.CORREL(abs(self.df.close), np.log(self.df.volume), timeperiod=window))
        return res

    def corr_pc_pv_3(self):
        window = 3
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res       

    def corr_pc_pv_5(self):
        window = 5
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res 

    def corr_pc_pv_8(self):
        window = 8
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res 

    def corr_pc_pv_13(self):
        window = 13
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res

    def corr_pc_pv_34(self):
        window = 34
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res

    def corr_pc_pv_55(self):
        window = 55
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res

    def corr_pc_pv_88(self):
        window = 88
        res = np.nan_to_num(talib.CORREL(self.df.close.pct_change().fillna(0), 
                                        self.df.volume.pct_change().fillna(0), 
                                        timeperiod=window))
        return res

    def up_days_pct_3(self):
        window = 3
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def up_days_pct_5(self):
        window = 5
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def up_days_pct_8(self):
        window = 8
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def up_days_pct_13(self):
        window = 13
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def up_days_pct_34(self):
        window = 34
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def up_days_pct_55(self):
        window = 55
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def up_days_pct_88(self):
        window = 88
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            up_days = up_days/window
            res.append(up_days)
        res = np.asarray(res)
        return res

    def down_days_pct_3(self):
        window = 3
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_days_pct_5(self):
        window = 5
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_days_pct_8(self):
        window = 8
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_days_pct_13(self):
        window = 13
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_days_pct_34(self):
        window = 34
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_days_pct_55(self):
        window = 55
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_days_pct_88(self):
        window = 88
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                else:
                    pass
            down_days = down_days/window
            res.append(down_days)
        res = np.asarray(res)
        return res

    def down_up_diff_3(self):
        window = 3
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

    def down_up_diff_5(self):
        window = 5
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

    def down_up_diff_8(self):
        window = 8
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

    def down_up_diff_13(self):
        window = 13
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

    def down_up_diff_34(self):
        window = 34
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

    def down_up_diff_55(self):
        window = 55
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

    def down_up_diff_88(self):
        window = 88
        window_1 = window+1
        res = [0]*window_1
        for i in range(len(self.df.close)-window_1):
            tmp_close = self.df.close[i:i+window_1]
            tmp_pct = tmp_close.pct_change().fillna(0)
            up_days = 0
            down_days = 0
            for a in range(len(tmp_pct)):
                tmp_pct_1 = tmp_pct.iloc[a]
                if tmp_pct_1 < 0:
                    down_days +=1
                elif tmp_pct_1 > 0:
                    up_days +=1
                else:
                    pass
            diff = up_days-down_days
            res.append(diff)
        res = np.asarray(res)
        return res

def get_factors(df):
        #print(0)
        factor = Alphas158(df)
        df1 = df.copy()
        #print(1)
        df1['kmid'] = factor.kmid()
        df1['klen'] = factor.klen()
        df1['kmid2'] = factor.kmid2()
        df1['kup'] = factor.kup()
        df1['kup2'] = factor.kup2()
        df1['klow'] = factor.klow()
        df1['klow2'] = factor.klow2()
        df1['ksft'] = factor.ksft()
        #print(2)
        df1['ksft2'] = factor.ksft2()
        df1['kmax_3'] = factor.kmax_3()
        df1['kmax_5'] = factor.kmax_5()
        df1['kmax_8'] = factor.kmax_8()
        df1['kmax_13'] = factor.kmax_13()
        df1['kmax_21'] = factor.kmax_21()
        #print(3)
        df1['kmax_34'] = factor.kmax_34()
        df1['kmax_55'] = factor.kmax_55()
        df1['kmax_88'] = factor.kmax_88()
        df1['klow_3'] = factor.klow_3()
        df1['klow_5'] = factor.klow_5()
        #print(4)
        df1['klow_8'] = factor.klow_8()
        df1['klow_13'] = factor.klow_13()
        df1['klow_34'] = factor.klow_34()
        df1['klow_55'] = factor.klow_55()
        df1['klow_88'] = factor.klow_88()
        #print(5)
        df1['rsv_3'] = factor.rsv_3()
        df1['rsv_5'] = factor.rsv_5()
        df1['rsv_9'] = factor.rsv_9()
        df1['rsv_13'] = factor.rsv_13()
        df1['rsv_34'] = factor.rsv_34()
        df1['rsv_55'] = factor.rsv_55()
        df1['rsv_88'] = factor.rsv_88()
        #print(6)
        df1['corr_c_log_v_3'] = factor.corr_c_log_v_3()
        df1['corr_c_log_v_5'] = factor.corr_c_log_v_5()
        df1['corr_c_log_v_8'] = factor.corr_c_log_v_8()
        df1['corr_c_log_v_13'] = factor.corr_c_log_v_13()
        df1['corr_c_log_v_34'] = factor.corr_c_log_v_34()
        df1['corr_c_log_v_55'] = factor.corr_c_log_v_55()
        df1['corr_c_log_v_88'] = factor.corr_c_log_v_88()
        #print(7)
        df1['corr_pc_pv_3'] = factor.corr_pc_pv_3()
        df1['corr_pc_pv_5'] = factor.corr_pc_pv_5()
        df1['corr_pc_pv_8'] = factor.corr_pc_pv_8()
        df1['corr_pc_pv_13'] = factor.corr_pc_pv_13()
        df1['corr_pc_pv_34'] = factor.corr_pc_pv_34()
        df1['corr_pc_pv_55'] = factor.corr_pc_pv_55()
        df1['corr_pc_pv_88'] = factor.corr_pc_pv_88()
        df1['up_days_pct_3'] = factor.up_days_pct_3()
        df1['up_days_pct_5'] = factor.up_days_pct_5()
        df1['up_days_pct_8'] = factor.up_days_pct_8()
        #print(8)
        df1['up_days_pct_13'] = factor.up_days_pct_13()
        df1['up_days_pct_34'] = factor.up_days_pct_34()
        df1['up_days_pct_55'] = factor.up_days_pct_55()
        df1['up_days_pct_88'] = factor.up_days_pct_88()
        df1['down_days_pct_3'] = factor.down_days_pct_3()
        df1['down_days_pct_5'] = factor.down_days_pct_5()
        df1['down_days_pct_8'] = factor.down_days_pct_8()
        df1['down_days_pct_13'] = factor.down_days_pct_13()
        df1['down_days_pct_34'] = factor.down_days_pct_34()
        df1['down_days_pct_55'] = factor.down_days_pct_55()
        df1['down_days_pct_88'] = factor.down_days_pct_88()
        #print(9)
        df1['down_up_diff_3'] = factor.down_up_diff_3()
        df1['down_up_diff_5'] = factor.down_up_diff_5()
        df1['down_up_diff_8'] = factor.down_up_diff_8()
        df1['down_up_diff_13'] = factor.down_up_diff_13()
        df1['down_up_diff_34'] = factor.down_up_diff_34()
        df1['down_up_diff_55'] = factor.down_up_diff_55()
        df1['down_up_diff_88'] = factor.down_up_diff_88()

        return df1


