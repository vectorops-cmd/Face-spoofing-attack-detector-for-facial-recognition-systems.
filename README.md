Face Spoofing Attack Detector for Facial Recognition Systems

How to run : 
Initiallly create an environment :  python -m venv venv
Then activate it : .\venv\Scripts\Activate.ps1
Install all requirements
and then run this command : python.app


📌 Overview

Face recognition systems are vulnerable to spoofing attacks such as printed photos, replayed videos, or screen-based impersonation. This mini project focuses on detecting face spoofing attacks to improve the reliability and security of facial recognition systems.

The system is designed as a web-based application where facial images are analyzed to distinguish between real (live) faces and spoofed (fake) faces using image-based features and machine learning techniques.

🎯 Objective

To build a face spoofing detection system that:

Identifies whether an input facial image is real or spoofed

Enhances security in facial authentication systems

Demonstrates practical integration of machine learning with a Flask backend

🛠️ Tech Stack

Frontend: HTML, CSS

Backend: Flask (Python)

Programming Languages: Python, Java

Dataset: CASIA Face Anti-Spoofing Dataset (CASIA FASD)

Data Type: Images

📂 Dataset

Name: CASIA Face Anti-Spoofing Dataset (CASIA FASD)

Source: Kaggle

Content:

Real face images

Spoofed face images (printed photos, video replays, etc.)

Usage: Training and testing the spoof detection model

⚙️ How the Project Runs
1️⃣ Activate Virtual Environment
cd C:\Users\Asus\Meghan07\mywebchatgpt
.\venv\Scripts\activate

2️⃣ Navigate to Backend Folder
cd backend

3️⃣ Run Flask Application
python app.py


Once the server starts, the web application can be accessed via the local browser.

🔍 Core Features Implemented

Image-based face spoof detection logic

Flask backend to handle requests

Basic frontend interface for interaction

Dataset preprocessing for facial images

Model integration (work in progress)

🚧 Project Status

Partially Completed

Implemented:

Backend structure using Flask

Dataset integration

Initial spoof detection workflow

Frontend layout

Pending:

Model performance optimization

Accuracy evaluation and metrics

Real-time webcam spoof detection

Improved UI/UX

Deployment

📉 Limitations

Works on static images only (no real-time detection yet)

Model accuracy not fully optimized

Limited evaluation metrics implemented

Requires manual dataset setup

🔮 Future Scope

Real-time spoof detection using webcam

Integration with deep learning models (CNNs)

Support for multiple spoofing attack types

Improved accuracy and performance benchmarking

Deployment on cloud platforms

👨‍💻 Project Type

Academic Mini Project
Developed as part of an engineering curriculum for learning purposes.
