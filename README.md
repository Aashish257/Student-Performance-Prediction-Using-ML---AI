# 🎓 Student Performance AI Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning application that predicts student academic performance based on demographic, social, and previous academic metrics. This project demonstrates full-stack ML engineering, from model integration to a modern, responsive user interface.

---

## ✨ Key Features

- **🎯 AI Predictions**: High-accuracy predictions using a Random Forest Classifier.
- **📱 Modern UI**: Responsive dashboard built with a Glassmorphism design system.
- **📊 Real-time Analysis**: Immediate feedback based on 13+ student metrics.
- **🔒 Secure Access**: Administrative login system for dashboard access.
- **🏗️ Clean Architecture**: Modularized backend with dedicated ML utilities and clean routing.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3 (Custom Glassmorphism System), Bootstrap 5, Jinja2
- **Data Visualization**: Chart.js (Planned)
- **Environment**: Virtualenv, Docker (Planned)

## 📐 Architecture

The project follows a modular structure for scalability and maintainability:

```
├── app.py              # Main Flask entry point
├── utils/
│   └── ml_model.py     # ML Model handler & Inference logic
├── static/
│   └── css/
│       └── style.css   # Custom Design System
├── template/           # Jinja2 HTML Templates
├── Dataset/            # UCI Student Performance Data
└── model.sav           # Trained RandomForest Serialized Model
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## 📊 Model Information

The model was trained on the **UCI Student Performance Dataset**, which includes student achievement in secondary education of two Portuguese schools.

**Input Features included:**
- `Demographics`: Age, Sex, Address Type
- `Family`: Parent education level, extra educational support
- `Social`: Travel time, frequency of going out, internet access
- `Academic`: Previous failures, Period 1 & 2 Grades

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
*Developed with ❤️ for Modern Education Technology.*
