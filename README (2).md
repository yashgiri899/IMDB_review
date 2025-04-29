# IMDB Movie Reviews Sentiment Analysis

This project focuses on performing **sentiment analysis** on the IMDB movie reviews dataset using **Natural Language Processing (NLP)** techniques and a **Neural Network model** built with **TensorFlow/Keras**.

---

## 🚀 Project Overview

- **Text Preprocessing** (lowercasing, removing punctuation, stopword removal, lemmatization)
- **TF-IDF Vectorization** for feature extraction
- **Neural Network** model for binary classification (positive/negative reviews)
- **Model Evaluation** using accuracy and classification report

---

## 📂 Dataset

The IMDB dataset used in this project contains **50,000 movie reviews** labeled as either positive or negative.

> **Note:** Due to file size limitations, the dataset is not uploaded in this repository. However, you can download it from the Kaggle link below:
>
> [🔗 IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

After downloading, place the dataset (`IMDB Dataset.csv`) inside a `data/` folder at the root of this project.


---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset and place it under:

```bash
./data/IMDB Dataset.csv
```


---

## 🧹 Project Structure

```
sentiment-analysis-imdb/
├── data/
│   └── IMDB Dataset.csv
├── train_model.py
├── requirements.txt
└── README.md
```


---

## 🚦 How to Run

Make sure you have placed the dataset correctly, then simply run:

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train the model
- Evaluate performance on the test set
- Print out the classification report


---

## 📈 Model Performance

The model is trained with EarlyStopping to avoid overfitting. Typical test accuracy is around **85-90%**, but it may vary slightly depending on training conditions.


---

## 📚 Libraries Used

- **pandas**
- **scikit-learn**
- **nltk**
- **tensorflow / keras**
- **numpy**


---

## 🤝 Contributing

Feel free to fork the project, make changes, and submit a pull request. Any suggestions for improvements are welcome!


---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).


---

## 🙌 Acknowledgements

Thanks to Kaggle and Lakshmi Narasimhan for providing the IMDB movie review dataset!

---

> Made with ❤️ for learning and exploring NLP and Deep Learning.
