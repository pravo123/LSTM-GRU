import os
import random
import threading
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

# Suppress TensorFlow INFO logs and oneDNN ops if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Seed for reproducibility
def set_seeds(seed=41):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()


class StockPredictor:
    def __init__(self):
        self.seq_length = 60
        self.model = None
        self.scaler = None

    def fetch_news_headlines(self, symbol):
        """Fetch news headlines or return default positive sentiment"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news and len(news) > 0:
                return [item.get('title', f"News about {symbol}") for item in news[:5]]
        except Exception:
            pass
        # Return default positive headlines if fetching fails
        return [
            f"Positive outlook for {symbol}",
            f"Investors optimistic on {symbol}",
            f"Economic indicators support {symbol}"
        ]

    def calculate_sentiment_score(self, symbol):
        """Calculate a simple sentiment score based on headlines"""
        headlines = self.fetch_news_headlines(symbol)
        # Simple sentiment calculation based on keywords
        positive_words = ['positive', 'optimistic', 'growth', 'increase', 'bull', 'rise', 'gain']
        negative_words = ['negative', 'pessimistic', 'decline', 'decrease', 'bear', 'fall', 'loss']

        total_score = 0
        for headline in headlines:
            headline_lower = headline.lower()
            pos_count = sum(1 for word in positive_words if word in headline_lower)
            neg_count = sum(1 for word in negative_words if word in headline_lower)
            total_score += (pos_count - neg_count) * 0.1

        return np.clip(total_score / len(headlines), -1, 1)

    def fetch_data(self, symbol, start_date, end_date, sentiment_score=None):
        """Fetch stock data and add features"""
        try:
            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
            spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=False, progress=False)

            if df.empty or spy.empty:
                raise ValueError("No data available for the given symbol or date range")

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.droplevel(1)

            df = df.copy()
            spy = spy.copy()

            df["Stock_Close"] = df["Close"]
            df["SPY_Close"] = spy["Close"]
            df["Sentiment"] = (
                sentiment_score
                if sentiment_score is not None
                else self.calculate_sentiment_score(symbol)
            )

            return self.add_features(df).fillna(method='ffill').dropna()

        except Exception as e:
            raise ValueError(f"Error fetching data: {str(e)}")

    def add_features(self, df):
        """Add technical indicators and features"""
        df = df.copy()
        df["Stock_Return"] = df["Stock_Close"].pct_change()
        df["SPY_Return"] = df["SPY_Close"].pct_change()
        df["SMA_9"] = df["Stock_Close"].rolling(window=9, min_periods=1).mean()
        df["SMA_49"] = df["Stock_Close"].rolling(window=49, min_periods=1).mean()
        df["EMA_9"] = df["Stock_Close"].ewm(span=9).mean()
        df["Momentum"] = df["Stock_Close"] - df["Stock_Close"].shift(9)
        df["Volatility"] = df["Stock_Return"].rolling(window=9, min_periods=1).std()
        return df

    def create_sequences(self, X, y):
        """Create sequences for LSTM/GRU training"""
        Xs, ys = [], []
        for i in range(len(X) - self.seq_length):
            Xs.append(X[i: i + self.seq_length])
            ys.append(y[i + self.seq_length])
        return np.array(Xs), np.array(ys)

    def prepare_data(self, df):
        """Prepare data for training"""
        features = [
            "Stock_Return", "SPY_Return",
            "SMA_9", "SMA_49",
            "EMA_9", "Momentum",
            "Volatility", "Sentiment"
        ]

        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        X = df[features].values
        y = df["Stock_Return"].values

        # Check for any remaining NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            # More aggressive cleaning
            df_clean = df[features + ["Stock_Return"]].dropna()
            X = df_clean[features].values
            y = df_clean["Stock_Return"].values

        if len(X) == 0:
            raise ValueError("No data remaining after cleaning")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")

        return train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    def build_model(self, input_shape, lstm_units, gru_units, dropout_rate):
        """Build the neural network model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            GRU(gru_units),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model = model
        return model

    def predict_future(self, df, days=30):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if self.scaler is None:
            raise ValueError("Scaler not initialized.")

        features = [
            "Stock_Return", "SPY_Return",
            "SMA_9", "SMA_49",
            "EMA_9", "Momentum",
            "Volatility", "Sentiment"
        ]

        X = self.scaler.transform(df[features].values)
        seq = X[-self.seq_length:]
        preds, price = [], df["Stock_Close"].iloc[-1]

        for _ in range(days):
            inp = seq.reshape(1, self.seq_length, seq.shape[1])
            ret = float(self.model.predict(inp, verbose=0)[0][0])
            price = price * (1 + ret)
            preds.append(price)

            # Create new sequence entry
            new_row = seq[-1].copy()
            new_row[0] = ret  # Update return
            seq = np.vstack([seq[1:], new_row])

        return preds


class ProgressCallback(Callback):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar

    def on_epoch_end(self, epoch, logs=None):
        if self.bar.winfo_exists():
            self.bar['value'] = epoch + 1
            self.bar.update()


class StockApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VolanX Hybrid Predictor")
        self.geometry("1200x900")

        # Set style
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use("clam")
        elif available_themes:
            style.theme_use(available_themes[0])

        self.predictor = StockPredictor()
        self.dark_mode = True
        self.create_widgets()
        self.set_defaults()

    def create_widgets(self):
        """Create the GUI widgets"""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Control panel
        ctrl = ttk.Frame(self, padding=10)
        ctrl.grid(row=0, column=0, sticky="ew")

        # Create input fields in a more organized layout
        row1 = ttk.Frame(ctrl)
        row1.pack(fill="x", pady=5)

        ttk.Label(row1, text="Symbol:").pack(side="left", padx=5)
        self.symbol = ttk.Entry(row1, width=10)
        self.symbol.pack(side="left", padx=5)

        ttk.Label(row1, text="Sentiment:").pack(side="left", padx=5)
        self.sent = ttk.Entry(row1, width=8)
        self.sent.pack(side="left", padx=5)

        ttk.Label(row1, text="Epochs:").pack(side="left", padx=5)
        self.epochs = ttk.Spinbox(row1, from_=1, to=200, width=8)
        self.epochs.pack(side="left", padx=5)

        ttk.Label(row1, text="Batch:").pack(side="left", padx=5)
        self.batch = ttk.Spinbox(row1, from_=1, to=256, width=8)
        self.batch.pack(side="left", padx=5)

        row2 = ttk.Frame(ctrl)
        row2.pack(fill="x", pady=5)

        ttk.Label(row2, text="LSTM units:").pack(side="left", padx=5)
        self.lstm_u = ttk.Spinbox(row2, from_=1, to=512, width=8)
        self.lstm_u.pack(side="left", padx=5)

        ttk.Label(row2, text="GRU units:").pack(side="left", padx=5)
        self.gru_u = ttk.Spinbox(row2, from_=1, to=512, width=8)
        self.gru_u.pack(side="left", padx=5)

        ttk.Label(row2, text="Dropout:").pack(side="left", padx=5)
        self.drop = ttk.Spinbox(row2, from_=0.0, to=0.9, increment=0.1, width=8)
        self.drop.pack(side="left", padx=5)

        # Buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill="x", pady=10)

        ttk.Button(btn_frame, text="Train & Predict", command=self.run_threaded).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Toggle Theme", command=self.toggle_theme).pack(side="left", padx=10)

        # Progress bar
        self.progress = ttk.Progressbar(self, length=600, maximum=1)
        self.progress.grid(row=1, column=0, pady=5)

        # Graph area
        graph = ttk.Frame(self, padding=10)
        graph.grid(row=2, column=0, sticky="nsew")
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Status bar
        stat = ttk.Frame(self, padding=10)
        stat.grid(row=3, column=0, sticky="ew")
        self.status = ttk.Label(stat, text="Ready", anchor="center")
        self.status.pack(fill="x")

    def set_defaults(self):
        """Set default values for input fields"""
        self.symbol.insert(0, "SPY")
        self.sent.insert(0, "0.10")
        self.epochs.set("50")
        self.batch.set("32")
        self.lstm_u.set("128")
        self.gru_u.set("64")
        self.drop.set("0.20")

    def toggle_theme(self):
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode
        style = ttk.Style()
        available_themes = style.theme_names()

        if self.dark_mode:
            if 'clam' in available_themes:
                style.theme_use("clam")
            bg, fg = "#2b2b2b", "#ffffff"
        else:
            if 'default' in available_themes:
                style.theme_use("default")
            bg, fg = "#ffffff", "#000000"

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.ax.tick_params(colors=fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)

        # Update legend if it exists
        legend = self.ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(bg)
            for text in legend.get_texts():
                text.set_color(fg)

        self.canvas.draw()

    def run_threaded(self):
        """Run the training and prediction in a separate thread"""
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()

    def run(self):
        """Main execution function"""
        try:
            # Get input values
            s = self.symbol.get().upper().strip()
            if not s:
                raise ValueError("Please enter a stock symbol")

            sent = float(self.sent.get())
            ep = int(self.epochs.get())
            bs = int(self.batch.get())
            lu = int(self.lstm_u.get())
            gu = int(self.gru_u.get())
            dr = float(self.drop.get())

            # Update status
            self.update_status("Fetching data...")
            df = self.predictor.fetch_data(
                s, "2020-01-01",  # Reduced date range for faster execution
                datetime.now().strftime("%Y-%m-%d"),
                sentiment_score=sent
            )

            self.update_status("Preparing data...")
            X_tr, X_te, y_tr, y_te = self.predictor.prepare_data(df)

            self.update_status("Building model...")
            self.predictor.build_model((self.predictor.seq_length, X_tr.shape[2]), lu, gu, dr)

            self.update_status("Training model...")
            self.progress['maximum'] = ep
            self.progress['value'] = 0

            cbs = [
                ProgressCallback(self.progress),
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
            ]

            self.predictor.model.fit(
                X_tr, y_tr, epochs=ep, batch_size=bs,
                validation_data=(X_te, y_te), callbacks=cbs, verbose=0
            )

            self.update_status("Predicting future...")
            preds = self.predictor.predict_future(df, days=30)
            self.plot_results(df, preds, s)
            self.update_status("Completed successfully!")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)  # Print to console for debugging
            self.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.update_status("Error occurred")

    def update_status(self, message):
        """Update status label from any thread"""
        self.after(0, lambda: self.status.config(text=message))

    def plot_results(self, df, preds, symbol):
        """Plot the results"""

        def do_plot():
            self.ax.clear()

            # Plot historical data (last 60 days)
            hist = df["Stock_Close"].iloc[-60:]

            # Create future dates
            last_date = df.index[-1]
            fut_dates = pd.bdate_range(start=last_date, periods=len(preds) + 1)[1:]

            # Plot
            self.ax.plot(hist.index, hist.values, label="Historical", linewidth=2, color='blue')
            self.ax.plot(fut_dates, preds, "--", label="Predicted", linewidth=2, color='red')

            # Styling
            self.ax.set_title(f"{symbol} Stock Price Prediction", fontsize=14, fontweight='bold')
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Price ($)")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)

            # Format x-axis
            self.ax.tick_params(axis='x', rotation=45)

            self.fig.tight_layout()
            self.canvas.draw()

            # Calculate percentage changes
            current_price = hist.iloc[-1]
            pct15 = (preds[14] / current_price - 1) * 100 if len(preds) > 14 else 0
            pct30 = (preds[-1] / current_price - 1) * 100

            self.status.config(text=f"Prediction: 15-day: {pct15:.2f}% | 30-day: {pct30:.2f}%")

        self.after(0, do_plot)


if __name__ == "__main__":
    app = StockApp()
    app.mainloop()