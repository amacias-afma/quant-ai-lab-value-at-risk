<!-- ```markdown -->
# Deep VaR: Physics-Informed Risk Estimation 📉🧠

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![Status](https://img.shields.io/badge/Status-Research_Complete-success)

> **"Don't throw away your GARCH model: Combine it with Deep Learning."**

## 📖 The Story: Why Standard AI Fails in Risk Management

Value at Risk (VaR) is the gold standard for risk measurement. However, estimating **99% VaR** (the 1% worst-case scenario) using Deep Learning is notoriously difficult.

In this project, I explore **why standard LSTMs fail** at extreme quantiles and how to fix them using a **Physics-Informed** approach.

### The Problem: The "Flat Loss" Trap
When training a Neural Network to predict a 1% probability event ($\alpha=0.01$):
1.  **Data Scarcity:** 99% of the data points provide weak gradient signals.
2.  **Instability:** The loss surface is flat, causing the optimizer to get "lost" or produce noisy, erratic predictions.
3.  **Benchmark Failure:** A complex LSTM often fails to beat a simple Historical Simulation or GARCH model (Occam's Razor).

### The Solution: Anchored Quantile Regression
Instead of letting the model guess wildly, I implemented a **Hybrid Loss Function**. I "anchor" the Neural Network to a robust statistical baseline (Parametric VaR).

$$\mathcal{L} = \underbrace{\text{PinballLoss}(y, \hat{y})}_{\text{Learn form Data}} + \lambda \cdot \underbrace{(\hat{y} - \text{Anchor})^2}_{\text{Guided by Theory}}$$

This gives the model the **stability** of classical statistics with the **adaptability** of Deep Learning.

---

## 📊 Results: Evolution of the Model

The research notebook documents the journey through three distinct phases:

| Phase | Model Type | Result | Diagnosis |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Naive LSTM** | ❌ **Failed** | Unstable, noisy predictions. Failed Kupiec POF test. |
| **Phase 2** | **Static Model** | ⚠️ **Fragile** | Good backtest, but failed during Regime Shifts (e.g., COVID crash). |
| **Phase 3** | **Anchored (Hybrid)** | ✅ **Success** | Beat the benchmark. Stable during calm periods, reactive during crashes. |

*(See `notebook/value_at_risk.ipynb` for the visualization of these phases)*

---

## 📂 Project Structure

```bash
quant-ai-lab-value-at-risk/
├── src/
│   ├── data/
│   │   └── market.py             # YFinance wrapper & data cleaning
│   ├── models/
│   │   └── deep_var/
│   │       ├── lstm_model.py     # PyTorch LSTM & Training Loop
│   │       ├── parametric_model.py # The "Anchor" (Benchmark calculations)
│   │       └── features.py       # Feature engineering (Rolling Variance, etc.)
│   ├── evaluation/
│   │   └── backtest_value_at_risk.py # Kupiec Test & Breach Statistics
│   └── utils/                    # Visualization & helper widgets
├── notebook/
│   └── value_at_risk.ipynb       # 📓 THE MAIN RESEARCH NARRATIVE
├── requirements.txt
└── README.md

```

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone [https://github.com/amacias-afma/quant-ai-lab-value-at-risk.git](https://github.com/amacias-afma/quant-ai-lab-value-at-risk.git)
cd quant-ai-lab-value-at-risk

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Research Notebook:**
Launch Jupyter Lab and open `notebook/value_at_risk.ipynb`. This notebook contains the full storytelling arc, training loops, and visualizations.
```bash
jupyter lab

```



---

## 🧠 Key Technical Concepts

* **Quantile Loss (Pinball Loss):** Asymmetric loss function for regression on specific percentiles.
* **Anchored Regularization:** Custom PyTorch loss implementation combining L1 (Quantile) and L2 (MSE vs Prior).
* **Walk-Forward Validation:** Preventing Look-Ahead Bias by strictly separating Train/Test chronologically.
* **Kupiec POF Test:** Statistical hypothesis testing to validate VaR breach rates.

---

## ⚖️ Disclaimer

*This project is for educational and research purposes only. It is not financial advice. Past performance of a model does not guarantee future results.*
