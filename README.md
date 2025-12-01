# MCQ Generator

**MCQ Generator** is a Flask-based web application that enables users to generate multiple-choice questions (MCQs) from text input. This tool is designed for educators, students, and content creators looking to automate quiz generation.

---

![Uploading Gemini_Generated_Image_3wahgu3wahgu3wah.png…]()



## ✨ Features

- ✅ Accepts input via URL, direct text entry, or file upload (PDF/TXT)
- ✅ Dynamically generates MCQs using NLP and deep learning
- ✅ Users can choose the number of questions to generate
- ✅ Displays questions in interactive quiz or detailed list formats
- ✅ Allows users to download questions as a PDF
- ✅ Simple, user-friendly UI with light/dark mode
- ✅ Responsive design powered by Bootstrap

---

## 🧰 Technologies Used

- **Python**
- **Flask** – Backend web framework
- **spaCy** – NLP processing and word embeddings
- **TensorFlow/Keras** – LSTM model for sentence structure learning
- **BeautifulSoup + Requests** – Web scraping for URL input
- **PyPDF2** – PDF text extraction
- **ReportLab** – PDF generation
- **Bootstrap** – Frontend UI framework
- **Gunicorn** – WSGI HTTP server for deployment

---

## 📁 Folder Structure

```
your_project_folder/
├── app.py              # Main Flask application logic
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── static/             # Static files (CSS, JS, images)
│   ├── style.css
│   └── main.js
└── templates/          # HTML templates
    ├── index.html
    ├── mcqs.html
    ├── result.html
    ├── 404.html
    └── 500.html
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder-name>
```

If the files are already present locally, simply navigate to the project folder.

### 2. Create a Virtual Environment

**On Windows:**

```bash
python -m venv venv
.env\Scriptsctivate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> _Note: Installing TensorFlow and spaCy may take some time._

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_md
```

---

## 🧪 Usage

### Run the Application

```bash
python app.py
```

Then visit `http://127.0.0.1:10000/` (or the port shown in your terminal).

### Generate MCQs

1. Choose input type: URL, text entry, or file upload (PDF/TXT)
2. Select the number of questions
3. Click **"Generate MCQs"**

### Interact with Questions

- Use the quiz view to test yourself
- Click **"Show Answers"** to reveal correct choices
- Switch to **List View** for a full breakdown
- Download the MCQs as a **PDF**
- Toggle between **light/dark themes** via the navbar

---

## 🌐 Deployment

To deploy on platforms like Heroku, Render, or Railway:

### Procfile

Create a `Procfile` (no extension) in the root directory:

```
web: gunicorn app:app
```

### Platform Configuration

**Build Command:**

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_md
```

**Start Command:**

```bash
gunicorn app:app
```

---

## ⚠️ Notes

- **Security:** Replace the default `app.secret_key` with a secure, randomly generated one. Use environment variables for production.
- **Model Performance:** The LSTM model is lightly trained and may not generalize well. For production, consider training on a large, domain-specific dataset.
- **Compatibility:** Maintain versions in `requirements.txt` to avoid conflicts, especially with TensorFlow and NumPy.
- **Error Handling:** Custom 404 and 500 pages are included, along with basic file size limits.

---

Happy Coding! 🚀
