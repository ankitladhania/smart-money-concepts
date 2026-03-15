[![PyPI](https://img.shields.io/pypi/v/smartmoneyconcepts.svg?style=flat-square)](https://pypi.org/project/smartmoneyconcepts/)
[![Downloads](https://pepy.tech/badge/smartmoneyconcepts/month)](https://pepy.tech/project/smartmoneyconcepts/month)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Bitcoin Donate](https://badgen.net/badge/Bitcoin/Donate/F19537?icon=bitcoin)](https://blockstream.info/address/bc1petss2mlqyjsajyzhu06wzl667v0f8svc0hnpqjj2d32frtx77g4sg5s0pg)

<p align="center">
  <img src="https://github.com/joshyattridge/smart-money-concepts/blob/f0c0fc28cc290cdd9dfcc6a6ac246ed1d59061be/tests/test.gif" alt="Candle Graph Showing Indicators"/>
</p>

# Smart Money Concepts (smc) (Causal)

The Smart Money Concepts Python Indicator is a sophisticated financial tool developed for traders and investors to gain insights into market sentiment, trends, and potential reversals. This indicator is inspired by Inner Circle Trader (ICT) concepts like Order blocks, Liquidity, Fair Value Gap, Swing Highs and Lows, Break of Structure, Change of Character, and more. Please Take a look and contribute to the project.

## Installation

```bash
pip install smartmoneyconcepts
```

## Usage

```python
from smartmoneyconcepts import smc
```

Prepare data to use with smc:

smc expects properly formated ohlc DataFrame, with column names in lowercase: ["open", "high", "low", "close"] and ["volume"] for indicators that expect ohlcv input.

## Causal Mode

All indicators that previously used lookahead (future data) now support a `causal=True` parameter. When enabled, outputs at bar N depend only on data from bars 0 through N — no future data leaks into past features. This is essential for ML/backtesting where lookahead bias must be eliminated.

```python
# Causal swing detection (lookback-only rolling windows)
swings = smc.swing_highs_lows(ohlc, swing_length=5, causal=True)

# All dependent functions accept causal=True and validate their inputs
fvg = smc.fvg(ohlc, causal=True)
bos_choch = smc.bos_choch(ohlc, swings, causal=True)
ob = smc.ob(ohlc, swings, causal=True)
liquidity = smc.liquidity(ohlc, swings, causal=True)
retracements = smc.retracements(ohlc, swings, causal=True)
```

**Key differences in causal mode:**
- `swing_highs_lows`: Last `swing_length` bars are NaN (unconfirmed). No synthetic endpoint.
- `fvg`: Last bar is NaN. Bar-by-bar mitigation tracking.
- `bos_choch`: Unbroken patterns kept with `BrokenIndex=NaN` (instead of removed).
- `ob`: Historical OBs are never erased by future price action.
- `liquidity`: Label placed at the confirmation bar (when 2nd swing joins group). `End` points back to the group start. Running pip_range instead of global.
- `retracements`: Non-wrapping shift for direction cleanup.
- All causal outputs set `result.attrs["causal"] = True` as metadata.
- Dependent functions (`bos_choch`, `ob`, `liquidity`, `retracements`) raise `ValueError` if called with `causal=True` but given non-causal swing inputs.

Default `causal=False` preserves all existing behavior exactly.

## Indicators

### Fair Value Gap (FVG)

```python
smc.fvg(ohlc, join_consecutive=False, causal=False)
```

A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
Or when the previous low is higher than the next high if the current candle is bearish.

parameters:<br>
join_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom<br>
causal: bool - if True then only past/present data is used (no lookahead)<br>

returns:<br>
FVG = 1 if bullish fair value gap, -1 if bearish fair value gap<br>
Top = the top of the fair value gap<br>
Bottom = the bottom of the fair value gap<br>
MitigatedIndex = the index of the candle that mitigated the fair value gap<br>

### Swing Highs and Lows

```python
smc.swing_highs_lows(ohlc, swing_length=50, causal=False)
```

A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

parameters:<br>
swing_length: int - the amount of candles to look back and forward to determine the swing high or low<br>
causal: bool - if True, uses lookback-only rolling windows. Last swing_length bars will be NaN (unconfirmed).<br>

returns:<br>
HighLow = 1 if swing high, -1 if swing low<br>
Level = the level of the swing high or low<br>

### Break of Structure (BOS) & Change of Character (CHoCH)

```python
smc.bos_choch(ohlc, swing_highs_lows, close_break=True, causal=False)
```

These are both indications of market structure changing

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.<br>
causal: bool - if True, uses bar-by-bar break validation. Requires causal swing inputs.<br>

returns:<br>
BOS = 1 if bullish break of structure, -1 if bearish break of structure<br>
CHOCH = 1 if bullish change of character, -1 if bearish change of character<br>
Level = the level of the break of structure or change of character<br>
BrokenIndex = the index of the candle that broke the level (NaN if not yet broken in causal mode)<br>

### Order Blocks (OB)

```python
smc.ob(ohlc, swing_highs_lows, close_mitigation=False, causal=False)
```

This method detects order blocks when there is a high amount of market orders exist on a price range.

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.<br>
causal: bool - if True, historical OBs are never retroactively erased. Requires causal swing inputs.<br>

returns:<br>
OB = 1 if bullish order block, -1 if bearish order block<br>
Top = top of the order block<br>
Bottom = bottom of the order block<br>
OBVolume = volume + 2 last volumes amounts<br>
Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume,lowVolume))<br>


### Liquidity

```python
smc.liquidity(ohlc, swing_highs_lows, range_percent=0.01, causal=False)
```

Liquidity is when there are multiple highs within a small range of each other,
or multiple lows within a small range of each other.

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
range_percent: float - the percentage of the range to determine liquidity<br>
causal: bool - if True, uses running pip_range and places labels at confirmation bar. Requires causal swing inputs.<br>

returns:<br>
Liquidity = 1 if bullish liquidity, -1 if bearish liquidity<br>
Level = the level of the liquidity<br>
End = the index of the last liquidity level (in causal mode: points back to group start)<br>
Swept = the index of the candle that swept the liquidity<br>

### Previous High And Low

```python
smc.previous_high_low(ohlc, time_frame="1D")
```

This method returns the previous high and low of the given time frame.

parameters:<br>
time_frame: str - the time frame to get the previous high and low 15m, 1H, 4H, 1D, 1W, 1M<br>

returns:<br>
PreviousHigh = the previous high<br>
PreviousLow = the previous low<br>
BrokenHigh = 1 once price has broken the previous high of the timeframe, 0 otherwise<br>
BrokenLow = 1 once price has broken the previous low of the timeframe, 0 otherwise<br>

### Sessions

```python
smc.sessions(ohlc, session, start_time, end_time, time_zone="UTC")
```

This method returns which candles are within the session specified

parameters:<br>
session: str - the session you want to check (Sydney, Tokyo, London, New York, Asian kill zone, London open kill zone, New York kill zone, london close kill zone, Custom)<br>
start_time: str - the start time of the session in the format "HH:MM" only required for custom session.<br>
end_time: str - the end time of the session in the format "HH:MM" only required for custom session.<br>
time_zone: str - the time zone of the candles can be in the format "UTC+0" or "GMT+0"<br>

returns:<br>
Active = 1 if the candle is within the session, 0 if not<br>
High = the highest point of the session<br>
Low = the lowest point of the session<br>

### Retracements

```python
smc.retracements(ohlc, swing_highs_lows, causal=False)
```

This method returns the percentage of a retracement from the swing high or low

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
causal: bool - if True, uses non-wrapping shift. Requires causal swing inputs.<br>

returns:<br>
Direction = 1 if bullish retracement, -1 if bearish retracement<br>
CurrentRetracement% = the current retracement percentage from the swing high or low<br>
DeepestRetracement% = the deepest retracement percentage from the swing high or low<br>

## Hide Credit Message

```bash
export SMC_CREDIT=0
```

This method will hide the credit message when you first import the library.

## Contributing

Please feel free to contribute to the project. By creating your own indicators or improving the existing ones. If you are struggling to find something to do then please check out the issues tab for requested changes.

1. Fork it (https://github.com/joshyattridge/smartmoneyconcepts/fork).
2. Study how it's implemented.
3. Create your feature branch (git checkout -b my-new-feature).
4. Commit your changes (git commit -am 'Add some feature').
5. Push to the branch (git push origin my-new-feature).
6. Create a new Pull Request.

Less is more – each pull request should be minimal, focusing on a single function or a small feature. Large, sweeping changes will not be merged, as they are harder to review and maintain. Keep it simple and focused!

## Disclaimer

This project is for educational purposes only. Do not use this indicator as a sole decision maker for your trades. Always use proper risk management and do your own research before making any trades. The author of this project is not responsible for any losses you may incur.
