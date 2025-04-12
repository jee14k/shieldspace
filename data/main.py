from fastapi import FastAPI
from pydantic import BaseModel
from model import display_analysis
import json


class Emotions(BaseModel):
    user_input: str


app = FastAPI()


@app.post("/emotions")
async def assess_emotions(emotion: Emotions):
    print("Received user input:", emotion)
    analysis = display_analysis(
        emotion.user_input)
    return {"analysis":  analysis}
