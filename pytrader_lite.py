import datetime as dt
import math
import os
from time import time

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from pandas.tseries.offsets import BDay, BMonthEnd
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import nolds  # for hurst exponent
import pandas_datareader.data as web
import requests_cache


class trader(object):

    def __init__(self, instrument, country="US", long_only=True,
                 num_periods=60, target_num_periods=20,
                 model_save_path="./models/"):

        self.country = country
        self.price_data = None
        self.sentiment_data = None
        self.fundamental_data = None
        self.econ_data = None
        self.model_save_path = model_save_path
        self.long_only = long_only
        self.all_returns = None
        self.price_changes = None
        self.num_periods = num_periods
        self.target_num_periods = target_num_periods

        self.instrument = instrument

        #training and testing vectors, predictions
        self.features = None
        self.labels = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.predictions = None

        #scores
        self.train_score = None
        self.test_score = None
        self.bm_train_score = None
        self.bm_test_score = None

        #returns due to predictions
        self.strategy_returns = None
        self.bm_returns = None

        #starting price level for q-learner
        self.prices = None
        self.train_start_price = 0.0
        self.test_start_price = 0.0

        #check if model_save dir exists, otherwise create it
        #self._ensure_dir(self.model_save_path)

        #scaler object set once train_test_split is run and can then be applied to test data
        self.scaler = None
        self.scaler_file_path = "./scaler.pkl"

    def _ensure_dir(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)


    def create_multi_period_features(self, periods):
        px_col = 'Adj Close'
        hi_col = 'High'
        lo_col = 'Low'

        if self.price_data is None:
            raise Exception("self.price_data is None, please run get_price_data first: {}".format(instrument))
        else:
            self.price_data['1d_ret'] = self.price_data[px_col].pct_change()
            #returns to create labels with, won't nead for learning
            self.all_returns = self.price_data['1d_ret'].copy()
            #price changes for q-learner rewards
            self.price_changes = self.price_data[px_col] - self.price_data[px_col].shift(1)
            #self.price_data['p_ret'] = self.price_data[px_col].pct_change(periods=self.num_periods)

            self.price_data['TARGET_RET'] = self.price_data[px_col].pct_change(periods=self.target_num_periods).shift(-self.target_num_periods)

            for period in periods:
                prefix = str(period) + "_day_"
                self.price_data[prefix + "ret"] = self.price_data[px_col].pct_change(periods=period)
                self.price_data[prefix + "vol"] = self.price_data['1d_ret'].ewm(halflife=period, min_periods=period).std() * math.sqrt(period)
                self.price_data[prefix + "sharpe"] = self.price_data[prefix + "ret"] / self.price_data[prefix + "vol"]
                self.price_data[prefix + 'dist_to_mov_avg'] = self.price_data[px_col] / self.price_data[px_col].ewm(halflife=period, min_periods=period).mean()
                self.price_data[prefix + 'skew'] = pd.Series.rolling(self.price_data['1d_ret'], window=period).skew() * (1/math.sqrt(period))
                self.price_data[prefix + 'kurt'] = pd.Series.rolling(self.price_data['1d_ret'], window=period).kurt() * (1/period)
                self.price_data[prefix + 'mom'] = self.price_data[prefix + "ret"].copy()
                #self.price_data[prefix + 'up_down_ratio'] = pd.Series.rolling(self.price_data[self.price_data['1d_ret'] >= 0]['1d_ret'], window=period).mean() / pd.Series.rolling(self.price_data[self.price_data['1d_ret'] < 0]['1d_ret'], window=period).mean()
                self.price_data.drop(str(prefix + "ret"), axis=1, inplace=True)

                #add the hurst exponent - period 5,10,20 outputs nonsenical data
                if period not in [5,10,20]:
                    hurst = lambda x: nolds.hurst_rs(x, fit="poly")
                    #print("Caluclating hurst...")
                    try:
                        self.price_data[prefix + 'hurst'] = pd.Series.rolling(self.price_data[px_col], 
                                                                              window=period).apply(hurst)
                    except:
                        self.price_data[prefix + 'hurst'] = 0.5


            self.price_data['macd'] = self._calc_MACD()
            self.price_data['rsi'] = self._calc_RSI()

            self.prices = self.price_data[px_col].copy()
            self.price_data.drop(['Open', 'High', 'Low', 
                        'Close', px_col, 'Volume', 
                        '1d_ret', '10_day_kurt', '5_day_skew',
                        '5_day_kurt', '10_day_skew'], 
                        axis=1, inplace=True)

            #print(self.price_data.shape)
                                   
            self.price_data.dropna(inplace=True)

  
    def _get_df_from_csv(self, filename):
        """Convenience function to load data from csv to dataframe and set index.

        Parameters
        ----------
        filename: the csv file to read

        returns: pandas DataFrame with Date as Index
        """
        df = pd.read_csv(filename)
        df.set_index('Date', drop=True, inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def _reindex_to_business_days(self, data):
        '''
        Convenience function to reindex a pandas dataframe to a full 5 business day week
        e.g. Mon - Fri
        Parameters:
        data: pandas DataFrame
        returns: reindexed DataFrame
        '''
        start_date = data.index[0]
        end_date = data.index[-1]
        date_range = pd.bdate_range(start_date, end_date)
        data = data.reindex(index=date_range, method='ffill', copy=True)
        data.dropna(inplace=True)
        return data

    def create_labels(self):
        go_short = -1
        if self.long_only is True:
            go_short = 0

        self.price_data['BUY_SELL'] = self.price_data['TARGET_RET']
        self.price_data['BUY_SELL'] = np.where(self.price_data['BUY_SELL'] < 0, go_short, 1)
        self.labels = self.price_data['BUY_SELL']
        self.price_data.drop('BUY_SELL', axis=1, inplace=True)

        try:
            self.price_data.drop('p_ret', axis=1, inplace=True)
        except:
            pass

        self.price_data.drop('TARGET_RET', axis=1, inplace=True)
        #print(self.price_data.head())
        #print(self.price_data.tail())
        self.features = self.price_data.copy()
        self.price_changes = self.price_changes.loc[self.features.index[0]:self.features.index[-1]].shift(-1)
        self.price_data = None
        #print(self.features.shape)

    def create_multi_period_features_and_labels(self, periods):
        self.create_multi_period_features(periods)
        self.create_labels()

    def scale_data(self, X=None):
        if self.scaler is None:
            self.scaler = self.load_scaler()
        else:
            if X is None:
                train_ix = self.X_train.index.copy()
                train_cols = self.X_train.columns.copy()

                test_ix = self.X_test.index.copy()
                test_cols = self.X_test.columns.copy()

                train_scaled = self.scaler.transform(self.X_train)
                test_scaled = self.scaler.transform(self.X_test)

                self.X_train = pd.DataFrame(data=train_scaled, index=train_ix, columns=train_cols)
                self.X_test = pd.DataFrame(data=test_scaled, index=test_ix, columns=test_cols)
            else:
                return self.scaler.transform(X)
    
    

    def train_classifier(self, clf, X, y):
        ''' Fits a classifier to the training data. '''

        # Start the clock, train the classifier, then stop the clock
        start = time()
        clf.fit(X, y.values)
        end = time()

        # Print the results
        diff = end-start
        if diff < 60:
            print ("Trained model in {:.2f} seconds".format(diff))
        elif diff < 3600:
            print("Trained model in {:.2f} minutes".format(diff/60))
        else:
            print("Trained model in {:.2f} hours".format(diff/3600))


    def predict_labels(self, clf, features, target, testing=False):
        ''' Makes predictions using a fit classifier based on F1 score. '''

        # Start the clock, make predictions, then stop the clock
        start = time()
        y_pred = clf.predict(features)
        end = time()

        # Print and return results
        diff = end-start
        #print ("Made predictions in {:.4f} seconds.".format(diff))
        if testing is True:
            self.predictions = y_pred
            self.bm_test_score = f1_score(target.values, 
                                          np.ones(len(target.values)), 
                                          pos_label=1)
            

        self.bm_train_score = f1_score(target.values, 
                                       np.ones(len(target.values)), 
                                       pos_label=1)
        

        return f1_score(target.values, y_pred, pos_label=1)


    def test_predict_new(self, clf, X_test, y_test):
        self.test_score = self.predict_labels(clf, X_test, y_test, testing=True)
        #print ("F1 score for {} for test set: {:.4f}.".format(clf.__class__.__name__, self.test_score))
        return self.test_score

    def create_strategy_from_predictions(self):
        test_start = pd.to_datetime(self.X_test.index[0])
        test_end = pd.to_datetime(self.X_test.index[-1])
        self.strategy_returns = self.all_returns.loc[test_start:test_end].shift(-1) * self.predictions
        self.strategy_returns.dropna(axis=0, inplace=True)
        self.bm_returns = self.all_returns.loc[test_start:test_end].copy()

    def load_scaler(self):
        try:
            self.scaler = joblib.load(self.scaler_file_path)
            #print("Scaler loaded successfully to self.scaler")
        except Exception as ex:
            print(ex)

  
    def save_scores(self, clf, security, perf_stats):
        dir = self.model_save_path + str(security) + "/"
        #self._ensure_dir(dir)
        filepath = dir + str(security) + "_" + str(self.num_periods) + "_day_" + clf.__class__.__name__ + "_scores.csv"
        data = { "train_f1": self.train_score,
                "test_f1": self.test_score,
                "train_bm_f1": self.bm_train_score,
                "test_bm_f1" : self.bm_test_score,
                "strategy_return_ann" : perf_stats.ann_perf,
                "bm_return_ann" : perf_stats.bm_ann_perf,
                "strategy_sharpe" : perf_stats.sharpe,
                "bm_sharpe" : perf_stats.bm_sharpe,
                "strategy_max_drawdown" : perf_stats.max_drawdown,
                "bm_max_drawdown" : perf_stats.bm_max_drawdown
                }
        df = pd.DataFrame(data=data, index=(security,))
        df = df[["train_f1", 
                 "train_bm_f1", 
                 "test_f1", 
                 "test_bm_f1", 
                 "strategy_return_ann", 
                 "bm_return_ann", 
                 "strategy_sharpe", 
                 "bm_sharpe", 
                 "strategy_max_drawdown", 
                 "bm_max_drawdown"]]
        #df.to_csv(filepath)
        #print("Saved scores to {}".format(filepath))
        return df

    
    def _calc_MACD(self):
        slow_ema = self.price_data['Adj Close'].ewm(span=26).mean()
        fast_ema = self.price_data['Adj Close'].ewm(span=12).mean()
        #self.price_data['MACD'] = fast_ema - slow_ema
        return (fast_ema - slow_ema)

    def _calc_RSI(self, n=14):
        deltas = self.price_data['Adj Close'].diff()
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up/down
        rsi = np.zeros_like(self.price_data['Adj Close'])
        rsi[:n] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(n, len(self.price_data['Adj Close'])):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up*(n-1)+upval)/n
            down = (down*(n-1)+downval)/n
            rs = up/down
            rsi[i] = 100.0 - 100.0/(1.0 + rs)

        return rsi

    def _calc_Williams_R(self, num_periods=60):
        px_col = 'Adj Close'
        hi_col = 'High'
        lo_col = 'Low'

        willr = ((self.price_data[hi_col].shift(-num_periods) - self.price_data[px_col]) / (self.price_data[hi_col].shift(-num_periods) - self.price_data[lo_col].shift(-num_periods))) * -100
        return willr

    def _calc_Momentum(self, num_periods):
        mom = self.price_data['Adj Close'].pct_change(periods=num_periods)
        return mom

    def _calc_ATR(self, num_periods):
        px_col = 'Adj Close'
        hi_col = 'High'
        lo_col = 'Low'

        tr1 = self.price_data[hi_col] - self.price_data[lo_col]
        tr2 = self.price_data[hi_col] - self.price_data[px_col].shift(1)
        tr3 = self.price_data[lo_col] - self.price_data[px_col].shift(1)
        df = pd.DataFrame(data=[tr1, tr2, tr3]).T
        atr = df.max(axis=1)
        atr = atr.ewm(halflife=num_periods).mean()
        return atr


class stats(object):

    def __init__(self, trader, year_day_count=252, risk_free=0):

        self.trader = trader
        self._daily_returns = self.trader.all_returns
        self._positions = self.trader.predictions
        self.year_day_count = year_day_count
        self.risk_free = risk_free
        self.num_days = len(self.daily_returns)

        self.strategy_returns = trader.strategy_returns
        self.bm_daily_returns = trader.bm_returns

        self.cum_performance = np.nan
        self.cum_volatility = np.nan
        self.sharpe = np.nan
        self.ann_perf = np.nan
        self.ann_vol = np.nan
        self.max_drawdown = np.nan

        self.bm_cum_performance = np.nan
        self.bm_cum_volatility = np.nan
        self.bm_sharpe = np.nan
        self.bm_ann_perf = np.nan
        self.bm_ann_vol = np.nan
        self.bm_max_drawdown = np.nan

        self.rel_cum_performance = np.nan
        self.rel_cum_volatility = np.nan
        self.rel_sharpe = np.nan
        self.rel_ann_perf = np.nan
        self.rel_ann_vol = np.nan
        self.rel_max_drawdown = np.nan

        self.rolling_cum_perf = np.nan
        self.rolling_bm_cum_perf = np.nan
        self.rolling_max_drawdown = np.nan
        self.rolling_bm_max_drawdown = np.nan

        self._calculated = False

    @property
    def daily_returns(self):
        return self._daily_returns

    @property
    def predicted_positions(self):
        return self._positions

    @daily_returns.setter
    def daily_returns(self, value):
        self._daily_returns = value

    @predicted_positions.setter
    def predicted_positions(self, value):
        self._positions = value

    def calculate(self):
        try: 
            #print(self.strategy_returns)
            self.cum_performance = np.prod(self.strategy_returns + 1) - 1
            self.cum_volatility = np.std(self.strategy_returns) * math.sqrt(len(self.strategy_returns))
            self.sharpe = self.cum_performance / self.cum_volatility

            self.bm_cum_performance = np.prod(self.bm_daily_returns + 1) - 1
            self.bm_cum_volatility = np.std(self.bm_daily_returns) * math.sqrt(len(self.bm_daily_returns))
            self.bm_sharpe = self.bm_cum_performance / self.bm_cum_volatility

            if len(self.strategy_returns) > self.year_day_count:
                #annualise the data, otherwise leave it
                self.ann_perf = math.pow((1 + self.cum_performance), 1 / (self.num_days / self.year_day_count)) - 1
                self.ann_vol = np.std(self.strategy_returns) * math.sqrt(self.year_day_count)
                self.sharpe = self.ann_perf / self.ann_vol

                self.bm_ann_perf = math.pow((1 + self.bm_cum_performance), 1 / (self.num_days / self.year_day_count)) - 1
                self.bm_ann_vol = np.std(self.bm_daily_returns) * math.sqrt(self.year_day_count)
                self.bm_sharpe = self.bm_ann_perf / self.bm_ann_vol
            else:
                self.ann_perf = self.cum_performance
                self.ann_vol = self.cum_volatility
                self.bm_ann_perf = self.bm_cum_performance
                self.bm_ann_vol = self.bm_cum_volatility

            self.rel_cum_performance = self.cum_performance - self.bm_cum_performance
            self.rel_cum_volatility = self.cum_volatility - self.bm_cum_volatility
            self.rel_sharpe = self.sharpe - self.bm_sharpe
            self.rel_ann_perf = self.ann_perf - self.bm_ann_perf
            self.rel_ann_vol = self.ann_vol - self.bm_ann_vol
            self.rel_max_drawdown = self.max_drawdown - self.bm_max_drawdown

            self.rolling_cum_perf = np.cumprod(self.strategy_returns + 1) - 1
            self.rolling_max_drawdown = self.rolling_cum_perf + 1
            self.rolling_max_drawdown = self.rolling_max_drawdown.div(self.rolling_max_drawdown.cummax()).sub(1)

            self.rolling_bm_cum_perf = np.cumprod(self.bm_daily_returns + 1) - 1
            self.rolling_bm_max_drawdown = self.rolling_bm_cum_perf + 1
            self.rolling_bm_max_drawdown = self.rolling_bm_max_drawdown.div(self.rolling_bm_max_drawdown.cummax()).sub(1)

            self.max_drawdown = self.rolling_max_drawdown.min()
            self.bm_max_drawdown = self.rolling_bm_max_drawdown.min()

            self._calculated = True
        except Exception as ex:
            print("Error calculating stats for {}".format(self.trader.instrument))

    def plot_tearsheet(self, instrument=None):

        if not self._calculated:
            print("Unable to plot tearsheet, calculate first")

        if instrument is None:
            instrument = self.trader.instrument

        ncols = 2
        nrows = 1

        ctr = 0

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, figsize=(12,4))
        fig.suptitle("{} strategy and drawdowns vs. benchmark".format(instrument), fontsize=12, y=1.05)

        for ax in axs.flat:
            if ctr == 0:
                self._plot_equity(self.rolling_cum_perf, self.rolling_bm_cum_perf, ax=ax)
            if ctr == 1:
                self._plot_drawdown(self.rolling_max_drawdown, self.rolling_bm_max_drawdown, ax=ax)
            ctr += 1
        
        #fig.subplots_adjust(top=1.2)
        plt.tight_layout()
        plt.show()

    def _plot_equity(self, ts, bm_ts, ax=None):

        def two_dec_format(x, pos):
            return "{:.0%}".format(x)

        equity = ts

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(two_dec_format)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        if bm_ts is not None:
            benchmark = bm_ts
            benchmark.plot(lw=2, color='grey', label="Benchmark", alpha=0.60, ax=ax)

        equity.plot(lw=2, color='xkcd:azure', alpha=0.6, x_compat=False, label='Strategy', ax=ax)
        ax.set_ylabel('Cumulative returns')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        return ax

    def _plot_drawdown(self, ts, bm_ts, ax=None):
        """
        Plots the underwater curve
        """
        def two_dec_format(x, pos):
            return "{:.0%}".format(x)

        equity = ts

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(two_dec_format)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        if bm_ts is not None:
            benchmark = bm_ts
            benchmark.plot(kind='area', lw=2, color='grey', label="Benchmark", alpha=0.60, ax=ax)

        equity.plot(kind='area', lw=2, color='xkcd:azure', alpha=0.6, x_compat=False, label='Strategy', ax=ax)
        ax.set_ylabel('Drawdown')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        return ax

   
class model_runner(object):

    def __init__(self, stock_tickers, model_filepath, start_date=dt.datetime(2017,3,31), end_date=dt.datetime.today()):
        """
        Convenience class to run pytrader predictions

        Parameters
        ----------
        stock_tickers: single ticker or list of tickers for use with Yahoo! Finance data
        start_date: data start date, default 31 March 2017
        end_date: data end date, default today.
        """
        self.trader = trader("MODEL_RUNNER")
        self.tickers = stock_tickers
        self.clf = self.load_model(model_filepath)
        self.trader.load_scaler()
        self.start_date = start_date
        self.end_date = end_date
        self.data_start_date = self.start_date - (120 * BDay()) # start 6 months prior
        self._session = requests_cache.CachedSession(cache_name='cache',
                                                     backend=None, expire_after=30)

        self.score_df = None

    def run(self):
        scores = []
        for ticker in self.tickers:
            try:             
                self.trader.price_data = web.DataReader(ticker, 
                                                        'yahoo', 
                                                        start=self.data_start_date, 
                                                        end=self.end_date, 
                                                        session=self._session)

                self.trader.price_data = self.trader._reindex_to_business_days(self.trader.price_data)
                #self.trader.price_data = self._get_local_data(instrument)
                if self.trader.price_data is not None:
                    self.trader.create_multi_period_features_and_labels([5,10,20,60,120])
                    self.trader.X_test = self.trader.features
                    self.trader.y_test = self.trader.labels
                    self.trader.X_train = self.trader.features
                    self.trader.y_train = self.trader.labels
                    self.trader.scale_data()
                    self.trader.test_predict_new(self.clf, self.trader.X_test, self.trader.y_test)
                    self.trader.create_strategy_from_predictions()
                    perf_stats = stats(self.trader)
                    perf_stats.calculate()
                    score = self.trader.save_scores(self.clf, ticker, perf_stats)
                    perf_stats.plot_tearsheet(ticker)
                    scores.append(score)
            except Exception as ex:
                print("[EXCEPTION]: {}, {}".format(ticker, ex))
                
        score_df = pd.concat(scores)
        self.score_df = self._get_score_diffs(score_df)
        

    def show_scores(self):
        display(self.score_df.style.format("{:.4f}"))

    def _get_local_data(self, instrument):
        path = './validation/{}_20171018.csv'.format(instrument)
        df = self.trader._get_df_from_csv(path) 
        return self.trader._reindex_to_business_days(df)

    def _get_data(self, ticker, start_date, end_date):
        try:
            df = web.DataReader(ticker, 'yahoo', start=start_date, end=end_date)
            #df.set_index('Date', inplace=True, drop=True)
            return self.trader._reindex_to_business_days(df)
        except Exception as ex:
            print("pandas_datareader failed to get data from Yahoo! Finance for ticker {}: {}".format(ticker, ex))
            return None

    def load_model(self, filepath):
        return joblib.load(filepath)

    def _get_score_diffs(self, df):
        diffs = pd.DataFrame()
        diffs['f1_diff'] = df.test_f1 - df.test_bm_f1
        diffs['return_diff'] = df.strategy_return_ann - df.bm_return_ann
        diffs['sharpe_diff'] = df.strategy_sharpe - df.bm_sharpe
        diffs['drawdown_diff'] = df.strategy_max_drawdown - df.bm_max_drawdown
        #display(diffs)
        return diffs
