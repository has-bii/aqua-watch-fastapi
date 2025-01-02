import os
from dotenv import load_dotenv
import datetime
from src.utils.predict import init_predict
from src.utils.predict_v2 import predict_v2
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load ENV
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Supabase client
supabase: Client = create_client(url, key)

# Init App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"Hello": url}

@app.get("/predict")
def get_auth(env_id: str):
    requested_date = datetime.datetime.now()
    requested_date_end = requested_date - datetime.timedelta(days=30)
    try:
        # Check if exists
        supabase.table("environment").select("*").eq("id", env_id).single().execute()
        
        # Supabase query
        response = supabase.table("dataset").select("*", count="exact").eq("env_id", env_id).gte("created_at", requested_date_end).csv().execute()
        
        if response.count == 0:
            return {"message": "Data is not sufficient"}
        
        predicted = init_predict(response.data)
        
        return {"data": predicted, "count": response.count}
        
    except Exception as error:
        print(error)
        raise HTTPException(status_code=404, detail="Internal server error")
    except:
        return {"message": "Internal server error2"}

@app.get("/predict/v2")
def get_auth(env_id: str):
    requested_date = datetime.datetime.now()
    requested_date_end = requested_date - datetime.timedelta(days=30)
    try:
        # Check if exists
        supabase.table("environment").select("*").eq("id", env_id).single().execute()
        
        # Supabase query
        response = supabase.table("dataset").select("*", count="exact").eq("env_id", env_id).gte("created_at", requested_date_end).order('created_at', desc=True).limit(2000).csv().execute()
        
        if response.count == 0:
            return {"message": "Data is not sufficient"}
        
        predicted = predict_v2(response.data)
        
        return {"data": predicted, "count": response.count}
        
    except Exception as error:
        print(error)
        raise HTTPException(status_code=404, detail="Internal server error")
    except:
        return {"message": "Internal server error2"}
