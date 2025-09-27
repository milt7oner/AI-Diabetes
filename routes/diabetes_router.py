from fastapi import APIRouter

from schemas.diabetes_schemas import PatientData
from services.diabletes_service import diabetes_prediction

router = APIRouter()

@router.post("/predict")
async def patientPredict(data: PatientData):
    print("Idetification user",data.identification)
    prediction = diabetes_prediction(data)
    return {"prediction": prediction}
