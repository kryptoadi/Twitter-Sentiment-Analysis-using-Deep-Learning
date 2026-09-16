# 🐦 Twitter Sentiment Analysis using RNN

<p align="center">
  <b>Deep Learning • NLP • RNN • Sentiment Classification</b>
</p>

<p align="center">
  A complete end-to-end Twitter sentiment analysis project using a<br>
  <b>Recurrent Neural Network (RNN)</b>, with training accelerated using the <b>Google Colab GPU</b>.
</p>

---

## 📌 Overview

This project uses **Natural Language Processing (NLP)** and a **Recurrent Neural Network (RNN)** to classify Twitter/X text into three sentiment categories:

- 🔴 **Negative**
- ⚪ **Neutral**
- 🟢 **Positive**

The project covers the complete workflow from **raw tweet data → text preprocessing → tokenization → RNN training → model saving → prediction/deployment**.

The deep-learning model was trained using the **Google Colab GPU** to accelerate neural-network training, while the project and deployment code are organized and developed through **VS Code**.

---

## ✨ Features

- 🧹 Text preprocessing and cleaning
- 🔤 NLP tokenization
- 🧠 RNN-based deep learning model
- 🎯 Three-class sentiment classification
- ⚡ GPU-accelerated model training using **Google Colab**
- 💾 Saved trained model (`model.h5`)
- 🔤 Saved tokenizer (`tokenizer.pkl`)
- 📊 Training and validation datasets
- 🚀 Prediction/deployment script using `main.py`
- 📦 Reproducible dependencies through `requirements.txt`

---

## 🧠 Model Architecture

The core of this project is a **Recurrent Neural Network (RNN)**.

An RNN is suitable for text because it processes sequences while carrying information from previous time steps.

### Basic flow

```text
                    Tweet / Text
                         │
                         ▼
                 Text Preprocessing
                         │
                         ▼
                    Tokenization
                         │
                         ▼
                    Input Sequence
                         │
                         ▼
                 ┌───────────────┐
                 │      RNN      │
                 └───────────────┘
                         │
                         ▼
                  Classification
                         │
                         ▼
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Negative     Neutral    Positive
```

During training, the RNN learns patterns in sequences of words/tokens that are associated with different sentiment classes.

---

## ⚡ GPU Training

The RNN model was trained using a **Google Colab GPU**.

Using a GPU is particularly useful for deep-learning workloads because neural-network operations involve large numbers of matrix/tensor calculations that can be parallelized.

### Training environment

| Component | Used |
|---|---|
| Development | VS Code |
| Notebook | Jupyter Notebook |
| Deep Learning | TensorFlow / Keras |
| Model | RNN |
| NLP | Tokenization & text preprocessing |
| Training Hardware | **Google Colab GPU** |
| Deployment | Python |
| Model File | `model.h5` |
| Tokenizer | `tokenizer.pkl` |

> **Note:** The GPU was used for model training; the saved model can subsequently be loaded for inference without needing to retrain it.

---

## 📂 Project Structure

```text
rnndeploy/
│
├── .vscode/
│
├── main.py
├── model.h5
├── tokenizer.pkl
├── requirements.txt
│
├── sentiment_analysis.ipynb
│
├── twitter_training.csv
├── twitter_validation.csv
│
└── README.md
```

### File description

| File | Purpose |
|---|---|
| `sentiment_analysis.ipynb` | Data preprocessing, model development and training |
| `twitter_training.csv` | Training dataset |
| `twitter_validation.csv` | Validation dataset |
| `model.h5` | Trained RNN model |
| `tokenizer.pkl` | Saved tokenizer used to convert text into model-compatible sequences |
| `main.py` | Prediction/deployment application |
| `requirements.txt` | Python dependencies |
| `.vscode/` | VS Code project configuration |

---

## 🔄 End-to-End Workflow

### 1. Data Loading

The Twitter datasets are loaded into Python using Pandas.

```python
import pandas as pd

tt = pd.read_csv("twitter_training.csv")
```

### 2. Text Preprocessing

Raw tweet text is cleaned before being passed to the neural network.

Typical NLP preprocessing can include:

- Removing unwanted characters
- Normalizing text
- Removing unnecessary whitespace
- Tokenization
- Converting text into numerical sequences

### 3. Tokenization

Text cannot be directly fed into a neural network.

The tokenizer converts words/tokens into numerical representations:

```text
Tweet
  ↓
Tokenization
  ↓
Numerical sequence
  ↓
RNN
```

The trained tokenizer is saved as:

```text
tokenizer.pkl
```

This allows the same token-to-index mapping to be reused during inference.

### 4. RNN Training

The processed sequences are supplied to the RNN.

The model learns the relationship between the input text and the sentiment labels:

```text
Text → Numerical Sequence → RNN → Sentiment
```

Training was accelerated using the **Google Colab GPU**.

### 5. Model Saving

After training, the model is saved as:

```text
model.h5
```

This means the trained model can be loaded later without training it again.

### 6. Prediction / Deployment

`main.py` loads the trained model and tokenizer and uses them to generate sentiment predictions.

The sentiment mapping used by the application is:

```python
sentiment_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
```

---

## 🛠️ Tech Stack

### Programming
- Python

### Machine Learning / Deep Learning
- TensorFlow
- Keras
- Recurrent Neural Networks (RNN)

### NLP
- Text preprocessing
- Tokenization
- Sequence processing

### Data Processing
- Pandas
- NumPy

### Development & Training
- VS Code
- Jupyter Notebook
- Google Colab GPU

### Deployment
- Python
- Saved Keras model
- Saved tokenizer

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd rnndeploy
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you are using the Google Colab GPU for training, install the project's compatible dependencies inside the Colab runtime rather than relying on the local Windows environment.

### 4. Run the application

```bash
python main.py
```

---

## 📊 Sentiment Classes

The model performs **multi-class classification**:

```text
                 Tweet
                   │
                   ▼
                 RNN
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Negative  Neutral  Positive
```

The output class is converted into a human-readable sentiment using the sentiment mapping in the application.

---

## 💡 Why RNN?

Text is sequential data.

The meaning of a word can depend on words that appeared before it. An RNN maintains a **hidden state** that carries information from previous time steps while processing the sequence.

For example:

```text
"I" → "really" → "love" → "this" → "movie"
```

The RNN processes these tokens sequentially and updates its internal hidden state as it moves through the sentence.

This makes RNNs a useful architecture for learning from sequential text.

---

## ⚠️ Limitations

A basic RNN can have difficulty learning dependencies across very long sequences because of issues such as:

- Vanishing gradients
- Exploding gradients
- Limited long-term memory

For larger or more complex NLP problems, architectures such as **LSTM, GRU, or Transformer-based models** can be considered.

---

## 🔮 Future Improvements

Possible extensions include:

- [ ] Compare RNN with LSTM and GRU
- [ ] Add more extensive text normalization
- [ ] Tune hyperparameters
- [ ] Add confusion matrix and classification metrics
- [ ] Track precision, recall and F1-score
- [ ] Experiment with pretrained word embeddings
- [ ] Deploy as a REST API
- [ ] Add a web interface for live tweet prediction
- [ ] Compare performance against a Transformer-based model

---

## 🎯 Learning Outcomes

Through this project, the following concepts are demonstrated:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Sequence representation
- Recurrent Neural Networks
- Forward propagation
- Backpropagation Through Time (BPTT)
- Sentiment classification
- GPU-accelerated deep-learning training
- Model serialization
- Model inference/deployment

---

## 👨‍💻 Project

**Twitter Sentiment Analysis using RNN**

Built as an end-to-end NLP and deep-learning project, from dataset preprocessing and RNN training to saving the trained model and using it for sentiment prediction.

---

<p align="center">
  <b>🐦 Turning tweets into sentiment with Deep Learning.</b>
</p>
