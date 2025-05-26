from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Legal Area of Law Classifier API")
app.include_router(router)
