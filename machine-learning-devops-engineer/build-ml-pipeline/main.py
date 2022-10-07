import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from build_ml_pipeline.utils import load, read_yaml
from build_ml_pipeline.model import process_data, do_inference

CONFIG_PATH = "./conf/config.yaml"
CONF = read_yaml(CONFIG_PATH)

# Instantiate the app.
app = FastAPI()

class ModelInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    # Write an example instance for the fastAPI docs.
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

# Retrieve trained artifacts:
MODEL, ENCODER, LAB_BIN = load(CONF["paths"]["model"])

@app.get("/")
async def run():
    # To query this api with get at the root domain, 
    # we would get back the following dict.
    return "Welcome to the app. Use the /predict to infer salary levels."

@app.post("/predict")
async def predict_salary(feature: ModelInput):

    feat, _, _, _ = process_data(
        pd.DataFrame(feature.dict(), index=[0]),
        CONF["data"]["cat_features"],
        training=False,
        encoder=ENCODER,
        lab_bin=LAB_BIN)

    pred = do_inference(MODEL, feat)
    pred = LAB_BIN.inverse_transform(pred)[0]

    return {"Predicted salary": pred}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=CONF["api"]["host"],
        port=CONF["api"]["port"],
        reload=True
    )
