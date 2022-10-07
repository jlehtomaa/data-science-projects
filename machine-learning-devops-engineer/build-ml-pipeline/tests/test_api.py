import json 
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get():

    resp = client.get("/")
    assert resp.status_code == 200

def test_high_salary():

    post = dict(
        age=50,
        workclass="Private",
        fnlgt=193524,
        education="Doctorate",
        education_num=16,
        marital_status="Married-civ-spouse",
        occupation="Prof-specialty",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=50,
        native_country="United-States"
    )
    resp = client.post("/predict", json=post)
    assert resp.status_code == 200
    assert resp.json() == {"Predicted salary": ">50K"}

def test_low_salary():

    post = dict(
        age=50,
        workclass="Self-emp-inc",
        fnlgt=346253,
        education="HS-grad",
        education_num=9,
        marital_status="Divorced",
        occupation="Sales",
        relationship="Own-child",
        race="White",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=30,
        native_country="United-States"
    )
    resp = client.post("/predict", json=post)
    assert resp.status_code == 200
    assert resp.json() == {"Predicted salary": "<=50K"}