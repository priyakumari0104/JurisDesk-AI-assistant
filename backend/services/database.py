import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["legal_ai"]
documents_collection = db["documents"]
print("MongoDb connection created")
try:
    client.admin.command("ping")
    print("MongoDb connected succesfully")
except Exception as e:
    print("MongoDb connection  failed:", e)
