# Pytrader
<div>
</div>

Pytrader is a small library making use of Scikit-learn's SVC, RandomForestClassifier and VotingClassifier classes to make predictions about the direction of stocks on the US stock exchanges.

The library presented here is the final version submitted for Udacity's Machine Learning Engineer Nanodegree.

The library therefore omits some of the functions that were used to train and test the model in development.

## Requirements
<ol>
<li> Python 3.4 or greater</li>
<li> pandas (0.20.3)</li>
<li>pandas-datareader (0.5.0)</li>
<li>scikit-learn (0.19.0)</li>
<li><a href="https://github.com/CSchoel/nolds">nolds</a></li>
<li>requests-cache-0.4.13</li>
</ol>

<hr>

## model_runner

A utility class to run the model on given stock tickers. 

model_runner makes use of the pandas_datareader Yahoo! Finance API. requests_cache is used in order to be a good web citizen, however Yahoo made some changes to the Yahoo! Finance website recently and whilst the pandas_datareader folks have done a fantastic job updating the library, it seems a little more fragile than previously.

> model_runner(stock_tickers, model_filepath, start_date, end_date)
<ul>
<li>stock_tickers: list of one or more tickers</li>
<li>model_filepath: path to model pkl</li>
<li>start_date: default 31 March 2017</li>
<li>end_date: default today</li>
</ul>

## Example usage

```python
runner = pytrader_lite.model_runner(list_of_tickers,
                                    path_to_model,
                                    '31-Mar-2017',
                                    '16-Oct-2017')
runner.run()
...
display(runner.score_df)
```

