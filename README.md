#  Legal Case Classification API (LawPavilion NLP Challenge)

This project is a Natural Language Processing (NLP) pipeline designed to classify legal case reports into their respective **areas of law** (e.g., Civil Procedure, Criminal Law, Company Law, etc.).

It supports inference via a **FastAPI endpoint** where users can submit the full body of a legal judgment (`full_report`) and receive the predicted legal category.

---

##  Project Objectives

- Clean and preprocess dataset
- Build a pipeline that maps legal case reports to legal categories.
- Compared traditional ML methods (TF-IDF + Logistic Regression) vs transformer-based deep learning models (DeBERTa).
- Serve the final transformer model as an API.

---

##  Dataset Overview

- **`full_report`**: The body of the legal judgment.
- **`introduction`**: Used to extract the area of law (labels).
- Other fields: `case_title`, `suitno`, `facts`, `issues`, `decision`.

Labels include:
- Civil Procedure
- Criminal Law and Procedure
- Enforcement of Fundamental Rights
- Company Law
- Election Petition, etc.

---

##  Approach

### 1 Traditional ML Pipeline

- **Text Vectorization**: TF-IDF (`TfidfVectorizer`)
- **Model**: Logistic Regression
- **Label Extraction**: Regex pattern matching from `introduction`
- **Evaluation**:
  - Accuracy: ~35%
  - F1 Score: Low for imbalanced classes
- **Limitation**: Unable to understand contextual meaning, struggled with nuanced legal phrasing.

### 2 Transformer-based Pipeline

- **Model Used**: `distilbert-base-uncased` fine-tuned on `full_report`
- **Tokenizer**: Hugging Face tokenizer
- **Label Encoder**: `LabelEncoder()` for mapping categories
- **Training**: Done on CPU (adapted for low-resource environment)
- **Evaluation**:
  - Higher F1 and accuracy
  - Better understanding of complex language
  - Handled rare labels more gracefully


---

## 🖥️ Project Structure

```
LawPavilion/
├── app/
│   ├── api.py             # FastAPI routes and logic
│   └── model_utilis.py    # Load model/tokenizer/label_encoder
├── models/
│   ├── config.json, tokenizer, model weights, saved_model.pkl
├── main.py                # Entrypoint for FastAPI app
├── requirements.txt
└── Legal_Classifier.ipynb # Jupyter notebook with traditional + BERT training
```

---

##  Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/shashacode/NLP_Code_Challenge.git
cd LawPavillion
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

Visit: `http://127.0.0.1:8000/docs` to access the Swagger UI.

---

## Example API Usage

**Endpoint**: `POST /predict`

**Request Body**:

```json
{
  "full_report": "The appellant was tried for armed robbery and sentenced to life imprisonment. The appeal concerns improper identification and denial of fair hearing."
}
```

**Response**:

```json
{
  "predicted_area_of_law": "Criminal Law and Procedure"
}
```

---



## Acknowledgment

This solution was developed as part of the **LawPavilion Legal NLP Challenge**, aimed at advancing AI solutions in the legal domain in Nigeria.
