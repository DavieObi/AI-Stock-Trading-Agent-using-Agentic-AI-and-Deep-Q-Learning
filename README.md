# AI Stock Trading Agent using Agentic AI and Deep Q-Learning

This repository contains an **AI trading agent** built using **Reinforcement Learning (Deep Q-Networks, DQN)** to simulate trading strategies for stocks. The agent interacts with a trading environment, learns optimal buy/sell/hold decisions from historical stock data, and attempts to maximize profits.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Methodology](#methodology)
* [Outputs](#outputs)
* [Notes & Warnings](#notes--warnings)
* [Future Improvements](#future-improvements)

---

## Overview

The AI agent leverages historical stock market data to **learn trading strategies** via a reinforcement learning framework. Using technical indicators and price movements, the agent makes sequential decisions to either **BUY, SELL, or HOLD** stocks.

The training process is based on **Deep Q-Learning**, a model-free reinforcement learning algorithm, which enables the agent to balance **exploration** (trying new actions) and **exploitation** (using learned strategies).

---

## Features

* Downloads historical stock data using [yfinance](https://pypi.org/project/yfinance/)
* Computes key technical indicators:

  * **Simple Moving Average (SMA)** for 5-day and 20-day periods
  * **Daily Returns**
* Defines a **custom trading environment** simulating account balance and holdings
* Implements a **Deep Q-Network (DQN)** agent with experience replay
* Trains the agent over multiple episodes to learn optimal trading strategies
* Tests the agent on unseen data to evaluate trading performance
* Outputs **total profit, final account balance**, and episode-wise rewards

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Install dependencies:

```bash
pip install yfinance pandas numpy torch
```

3. Run the main script:

```bash
python agentic_trading.py
```

---

## Usage

1. **Define stock and time period:**

```python
symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-02-14"
```

2. **Training the agent:**

The agent is trained for a predefined number of episodes (`episodes = 500`) to maximize profits:

```python
for episode in range(episodes):
    ...
```

3. **Testing the agent:**

After training, the agent is evaluated on the same dataset (or can be extended to unseen data):

```python
test_env = TradingEnvironment(data)
state = test_env.reset()
done = False
while not done:
    action = agent.act(state)  # exploit learned strategy
    next_state, reward, done, _ = test_env.step(action)
    state = next_state if next_state is not None else state
```

---

## Methodology

1. **Data Preprocessing:**

   * Historical stock prices are downloaded using `yfinance`.
   * Technical indicators (SMA 5, SMA 20, returns) are calculated.
   * Missing values are removed.

2. **State Representation:**
   Each state includes:

   * Current closing price
   * 5-day SMA
   * 20-day SMA
   * Daily returns

3. **Trading Environment:**

   * Initializes with an account balance ($10,000) and zero holdings.
   * Actions: `0=HOLD`, `1=BUY`, `2=SELL`.
   * Rewards are computed based on net profit at the end of each episode.

4. **DQN Agent:**

   * Neural network with 3 layers (64 neurons each hidden layer)
   * Experience replay buffer to store past experiences
   * Exploration-exploitation strategy controlled by epsilon decay
   * Loss function: Mean Squared Error (MSE)
   * Optimizer: Adam

5. **Training Loop:**

   * Agent interacts with environment and stores transitions `(state, action, reward, next_state, done)`
   * Mini-batch random sampling is used for model updates
   * Epsilon decay ensures the agent gradually favors exploitation over exploration

---

## Outputs

* **Episode-wise reward tracking:** Helps visualize learning progress over 500 episodes

Example of training outputs:

```
Episode 1/500, Total Reward: -9894.20
Episode 6/500, Total Reward: 5394.80
Episode 9/500, Total Reward: 6059.93
...
Episode 500/500, Total Reward: 11273.78
```

* **Testing results:** Evaluates performance after training:

```
Final Balance after testing: $14408.77
Total Profit: $4408.77
```

* The output demonstrates the agent's ability to increase account balance and generate profit based on learned strategies.

---

## Notes & Warnings

* `FutureWarning` messages appear due to using `float()` on single-element pandas Series. Use `float(series.iloc[0])` for future-proof code.
* Profits vary between runs due to stochastic exploration in DQN.
* The model currently trades on historical data; **past performance does not guarantee future profits**.

---

## Future Improvements

* Incorporate additional technical indicators (e.g., RSI, MACD)
* Implement **multi-stock trading** and portfolio optimization
* Use **longer training periods and more episodes** for better strategy learning
* Include visualization of trades and equity curve
* Experiment with **Double DQN or Dueling DQN** for improved performance

---
