## 📰 Fake News Classifier using Machine Learning

This project is a simple yet effective **Fake News Classifier** built using Python and Machine Learning. It takes in a **news headline** and predicts whether it is **Real** or **Fake**, based on training from actual news datasets.

---

###  Features

* Classifies news headlines as **Real** ✅ or **Fake** ❌.
* Trained using a **TF-IDF Vectorizer** and **PassiveAggressiveClassifier**.
* Evaluates model performance with **accuracy score** and **confusion matrix**.
* Runs interactively in the terminal for real-time predictions.

---

###  Tech Stack

* **Python**
* **Scikit-learn**
* **NLTK**
* **Pandas**, **NumPy**
* **TF-IDF** for text feature extraction

---

###  Dependencies

Make sure you have Python installed. Then, install these packages:

```bash
pip install pandas numpy scikit-learn nltk
```

Also, download NLTK stopwords (only once):

```python
import nltk
nltk.download('stopwords')
```

---

###  Dataset

The model uses:

* [`Fake.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* [`True.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

> Place both files in the correct path or update the path in the code accordingly.

---

### How to Run

1. Clone or download the project.
2. Update the file paths to your CSV files if needed.
3. Run the script:

```bash
python fake_news_classifier.py
```

4. You’ll see:

```
📰 Fake News Classifier is ready!
Type a news headline to classify it. Type 'q' to quit.
```

5. Enter headlines like:

```
WHO approves new malaria vaccine for global rollout
=> Prediction: ✅ Real
```

---

### 📊 Sample Output

```
✅ Model Accuracy: 95.30%
🧾 Confusion Matrix:
[[478  22]
 [ 18 482]]

📰 Fake News Classifier is ready!
Enter news headline: NASA to open space hotel by 2025
=> Prediction: ❌ Fake
```

screenshot:
![Image](https://github.com/user-attachments/assets/16d5d202-1fed-4676-931b-2f8532993639)
