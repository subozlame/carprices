🚗 Car Price Prediction — Machine Learning Project

This project predicts car prices using Machine Learning techniques in Python.
It uses a dataset of car details and trains a regression model to estimate the selling price based on input features.


📂 **Project Files**

| File Name        | Description                                           |
|------------------|-------------------------------------------------------|
| Cardetails.csv   | Dataset containing car information                    |
| app.py           | Streamlit web application                             |
| dataset.ipynb    | Jupyter notebook for data analysis and model training |
| requirements.txt | Download the requirements for the program             |

---

📂 **File Structure**

```
Project_Title/
│
├── venv/
│   └── ...
├── app.py
├── Cardetails.csv
├── dataset.ipynb
├── model.pkl                # Auto-generated during the process
├── requirements.txt         # Contains the requirements
└── run.txt                  # Run the app using Streamlit
```


📦 Virtual Environment Setup (env)

Follow these steps to set up your environment.

1️⃣ Create a virtual environment named env

=> Windows
```
python -m venv env
```

=> Linux/Mac

```
python3 -m venv env
```

2️⃣ Activate the virtual environment

=> Windows
```
env\Scripts\activate
```

=> Linux/Mac
```
source env/bin/activate
```


🛠 Requirements

The project uses the following Python libraries:

1. numpy
2. pandas
3. sklearn


📥 Install the requirements

Once the virtual environment is activated, install the required libraries:

```
pip install numpy pandas scikit-learn streamlit
```

Or if you have a requirements.txt, run:

```
pip install -r requirements.txt
```


▶️ Running the Jupyter Notebook

To explore the data and model training:

```
jupyter notebook dataset.ipynb
```


🌐 Run the Streamlit App (Localhost)

Use Streamlit to launch the prediction web app:

```
streamlit run app.py
```

This will open the app in your browser, usually at:

```
http://localhost:8501
```


🧠 How the Model Works

1. Load dataset (CarDetails.csv)

2. Clean & preprocess data

3. Train a machine learning model (Linear Regression / Random Forest / etc.)

4. Save or load the trained model in Streamlit

5. User inputs car details and gets predicted price


📝 Notes

- Ensure the dataset file (CarDetails.csv) is in the same directory as app.py

- Activate the virtual environment every time before running the app

- Use latest Python version (3.8 or above recommended)



