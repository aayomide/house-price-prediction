import uvicorn
from fastapi import FastAPI
from house import House
import pickle

app = FastAPI(title='Deploying a ML House Price Prediction Model with FastAPI')

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected."


@app.post('/predict')
def predict_house_price(house: House):
    bedrooms = house.bedrooms
    bathrooms = house.bathrooms
    sqft_living =house.sqft_living
    sqft_lot = house.sqft_lot
    floors = house.floors
    waterfront = house.waterfront
    view = house.view
    condition = house.condition
    grade = house.grade
    sqft_above = house.sqft_above
    sqft_basement = house.sqft_basement
    yr_built = house.yr_built
    yr_renovated = house.yr_renovated
    zipcode = house.zipcode
    lat =house.lat
    long =house.long
    sqft_living15 =house.sqft_living15
    sqft_lot15 =house.sqft_lot15
    year = house.date.year
    month = house.date.month
    day = house.date.day

    
    data = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, 
            sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, 
            sqft_lot15, day, month, year ]]

    #print(data)
    loaded_model = pickle.load(open('house_price_model.pkl', 'rb'))   # load saved model
    prediction = loaded_model.predict(data)

    return {
        "prediction": prediction.tolist()
        }

#run
if __name__ == '__main__':
    uvicorn.run(app)
