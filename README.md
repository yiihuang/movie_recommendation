# 🎬 Movie Recommendation System: Predicting User Preferences

This project addresses the problem of accurately predicting user movie preferences in the context of sparse rating data, a common challenge in recommendation systems. The team implemented and compared four collaborative filtering approaches using the MovieLens dataset: Neighborhood-Based Collaborative Filtering, Ridge Regression, Neural Networks, and Support Vector Machines. Each method was tailored to handle data sparsity and assessed based on prediction quality, computational efficiency, and system applicability. The models were evaluated using metrics like mean squared error, expected percentile ranking, and classification accuracy to determine their effectiveness in recommending movies.

# 📁 Project Structure

```
📁 movie-recommendation-system/
│
├── 📄 README.md                          # Project overview and setup instructions
├── requirements.txt                   # Python dependencies
│
├── 📂 data/                              # Directory for storing datasets (e.g., MovieLens)
│
├── create_data_matrix.ipynb          # Jupyter notebook for preprocessing and building data matrix
│
├── cf.py                              # Neighborhood-based Collaborative Filtering implementation
├── ridge.py                           # Ridge Regression-based Collaborative Filtering implementation
├── nn.py                              # Neural Network model definition for rating prediction
├── train_nn.py                        # Script for training the neural network
├── svm.py                             # Support Vector Machine approach for recommendations
├── util.py                            # Shared utility functions (e.g., similarity metrics, evaluation)
```
# 📊 Dataset Metadata (ml-latest-small)

## 📄 Summary

This dataset describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100,004 ratings and 1,296 tag applications across 9,125 movies. These data were collected from 671 users between January 09, 1995 and October 16, 2016. The dataset was generated on October 17, 2016.

* Each user rated at least 20 movies.
* Users were selected randomly and anonymized.
* No demographic information is included.

**Files included:**

* `links.csv`
* `movies.csv`
* `ratings.csv`
* `tags.csv`

⚠️ *Note: This is a development dataset, and may change. For publication-ready benchmarks, use official benchmark datasets.*

## 📁 File Descriptions

### 🧾 `ratings.csv`

* **Format:** `userId,movieId,rating,timestamp`
* **Scale:** 0.5 to 5.0 (in 0.5 steps)
* **Notes:** Ordered by userId, then movieId

### 🏷️ `tags.csv`

* **Format:** `userId,movieId,tag,timestamp`
* **Notes:** User-generated metadata; short phrases or words

### 🎞️ `movies.csv`

* **Format:** `movieId,title,genres`
* **Genres:** Pipe-separated, e.g., `Comedy|Drama|Romance`
* **Notes:** Titles include release year

### 🔗 `links.csv`

* **Format:** `movieId,imdbId,tmdbId`
* **Use:** Link MovieLens movies to IMDB and TMDB

## 🔧 Setup and Installation Instructions

**Important note:** please run this project on the server for the sake of version consistency.

### Step 1: Download Project Files

Run the following command in your local terminal to download the project:

```bash

```
### Step 2: Setup Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

Install necessary Python packages:

```bash
pip install -r requirements.txt
```

# 🚀 Step-by-Step Guide

## 🗂️ File Execution Order & Purpose

### 1. `create_data_matrix.ipynb`

**Purpose:**

* Preprocess raw MovieLens data and build user-item rating matrices.


Use Jupyter to manually execute cells.

---

### 2. `cf.py`

**Purpose:**

* Implements Neighborhood-based Collaborative Filtering using similarity metrics (e.g., cosine, Pearson).

**How to Run:**

```bash
python cf.py
```

Adjust internal parameters as needed (e.g., similarity type, top-K neighbors).

---

### 3. `ridge.py`

**Purpose:**

* Applies Ridge Regression to estimate user ratings using a linear model and regularization.

**How to Run:**

```bash
python ridge.py
```

You may need to install `scikit-learn` if not already available.

---

### 4. `nn.py`

**Purpose:**

* Defines the architecture for the neural network used in rating prediction.
* Used as a module, not directly executed.

**No command needed.**

---

### 5. `train_nn.py`

**Purpose:**

* Trains the neural network using user-item matrix data.
* Utilizes the model defined in `nn.py`.

**How to Run:**

```bash
python train_nn.py
```

Ensure PyTorch is installed:

```bash
pip install torch torchvision
```

---

### 6. `svm.py`

**Purpose:**

* Uses Support Vector Machines to predict ratings.

**How to Run:**

```bash
python svm.py
```

Install dependencies if needed:

```bash
pip install scikit-learn
```

---

### 7. `util.py`

**Purpose:**

* Contains utility functions shared across models (e.g., similarity calculation, evaluation metrics).

**No command needed.** Used as a helper module by other scripts.

---

## 📌 Notes

* All data files should reside in the `data/` directory.
* Execute models independently to compare results.
* Output predictions or evaluation results will typically be printed to console or saved as files (you may customize).

Feel free to adapt file parameters and code structures to fine-tune model performance.
