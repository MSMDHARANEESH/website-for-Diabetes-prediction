from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize FastAPI
app = FastAPI()

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Load the dataset
df = pd.read_csv(r'C:\Users\Dharaneesh S\OneDrive\Desktop\knn\diabetes.csv')

# Split features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Define route for the prediction page
@app.post("/predict/", response_class=HTMLResponse)
async def predict_diabetes(request: Request, Pregnancies: float = Form(...), Glucose: float = Form(...), BloodPressure: float = Form(...), SkinThickness: float = Form(...), Insulin: float = Form(...), BMI: float = Form(...), DiabetesPedigreeFunction: float = Form(...), Age: float = Form(...)):
    # Preprocess input data
    input_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Make prediction
    prediction = knn.predict(input_data)
    
    # Format prediction result
    if prediction[0] == 0:
        result = "Not affected by diabetes"
    else:
        result = "Affected by diabetes"

    # Render prediction result in HTML template
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# Define route for the input page
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

    



