Title: PROJETC FINANCIAL MARKETS - PAIR TRADING  ( FOCUS ON BACKTESTING)

Data Extraction :yfinance

Problem, Metedology and Results: 

Old                                                       | What You Want
--------------------------------------------------------- | -------------------------------------------------------------
Model decides Long / Short based on the Z-Score           | The spread band gives the trigger, but the model validates whether it’s a good moment
The model decides the signal alone                      | The model serves as a validator of trade quality (filter)


>> The Role of Classical Reversion (Bands)  
    The Z-Score or Spread touching the bands is a classic mean-reversion rule.

>> But we know that:  
    - Sometimes reversion happens.  
    - Other times the spread continues to diverge (breakouts, regime shifts) → your pair is not stationary.

>> Real Problem:  
    The bands (DP, Extra DP) give good signals in a stable regime → but they produce many false positives in trending/high volatility regimes.

>> Where the Model (RF) Helps:  
    The model acts as a "contextualizer" or "quality filter":

What It Does                                        | Why It’s Useful
--------------------------------------------------- | ---------------------------------------------------------
Evaluates the context of the spread at the moment   | Ex: Is the spread in a reversion regime or in a breakout?
Considers features like correlation, volatility, momentum, beta | Prevents entering trades with a low probability of reversion
Acts as a "stop-filter": only validates high-quality trades | Reduces false signals

Note: The model does NOT replace the bands → For your strategy, the bands remain the trigger, and the model validates the quality of the trade to decide if it’s worthwhile to enter or not.

The model only answers:  
➔ “Is it worth entering NOW?” (1: yes / 0: no).


Without Model                                      | With Model (RF Filter)
--------------------------------------------------- | ---------------------------------------------------------
Band touches → always enter                        | Band touches → enter only if the context supports it
Many false signals                                 | Fewer false signals, higher quality signals
Blind bet on reversion                             | Informed bet with context