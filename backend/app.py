
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
superkart_api = Flask("superkart_sales_api")
CORS(superkart_api)

# Load trained model
MODEL_PATH = "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully")
print("Expected model features:")


# Health check route
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API"

# Prediction route
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Parse JSON payload
        data = request.get_json()
        print("Raw incoming data:", data)

        # Validate expected fields
        required_fields = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_Type',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Product_Type_Category'
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f"Missing fields: {missing_fields}"}), 400

        # Convert and transform input
        sample = {
            'Product_Weight': float(data['Product_Weight']),
            'Product_Sugar_Content': data['Product_Sugar_Content'],
            'Product_Allocated_Area': float(data['Product_Allocated_Area']),
            'Product_Type': data['Product_Type'],
            'Product_MRP': float(data['Product_MRP']),
            'Store_Size': data['Store_Size'],
            'Store_Location_City_Type': data['Store_Location_City_Type'],
            'Store_Type': data['Store_Type'],
            'Product_Type_Category': data['Product_Type_Category']
        }

        input_df = pd.DataFrame([sample])
        print("Transformed input for model:\n", input_df)

        # Make sure columns are in exactly the same order
        input_df = input_df[model.feature_names_in_]

        print("Input sent to model:")
        print(input_df)

        # Make prediction
        prediction = model.predict(input_df).tolist()[0]
        print("Prediction:", prediction)

        return jsonify({
            'Sales': float(prediction)


        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Run the app (for local testing only)
if __name__ == '__main__':
    superkart_api.run(debug=True)
