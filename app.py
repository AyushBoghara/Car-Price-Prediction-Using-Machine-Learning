import pandas as pd
import joblib

# Load saved files
model = joblib.load("model/car_price_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# df_features = pd.DataFrame(feature_columns, columns=['Feature_Name'])
# df_features.to_csv(r'D:\Car Price Prediction\output_data.csv', index=False)
# print("Feature columns successfully saved!")

def predict_car_price(name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats):
    # Create input DataFrame
    car = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage(km/ltr/kg)": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }])

    # Convert numeric columns
    numeric_cols = [
        "year", "km_driven", "mileage(km/ltr/kg)",
        "engine", "max_power", "seats"
    ]

    for col in numeric_cols:
        car[col] = pd.to_numeric(car[col], errors="coerce")

    # Calculate car age using the same reference year
    car["age"] = 2026 - car["year"]

    # Ordinal Encoding and One-hot encode categorical columns
    car = pd.get_dummies(
        car,
        columns=[
            "name", "fuel", "seller_type",
            "transmission", "owner"
        ]
    )

    # Match training features and their exact order
    car = car.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Reject invalid numeric input
    if car.isna().any().any():
        raise ValueError("Please enter valid numeric values.")

    # Predict selling price
    predicted_price = model.predict(car)
    
    #  Output the result
    print(f"Predicted Car Price: INR {predicted_price[0]:,.2f}")


# Example prediction
predict_car_price(
    name="Hyundai i20 Asta 1.4 CRDi",
    year=2015,
    km_driven=225596,
    fuel="Diesel",
    seller_type="Individual",
    transmission="Manual",
    owner="Second Owner",
    mileage=20,
    engine=1396.0,
    max_power=88.76,
    seats=5
)
