from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
#uvicorn Search_data:app --reload
app = FastAPI()
class SearchItem(BaseModel):
    field: str
    value: str
# 設置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有方法
    allow_headers=["*"],  # 允許所有標頭
)

@app.get("/get_csv_data")
async def read_csv():
    # 讀取特定欄位
    data = pd.read_csv('Database.csv', usecols=['num', 'QA', 'Category', 'Filename'])
    # 返回 JSON
    return JSONResponse(content=data.to_dict(orient='records'))

@app.post("/search_csv_data")
async def search_csv_data(item: SearchItem):
    data = pd.read_csv('Database.csv', usecols=['num', 'QA', 'Category', 'Filename'])
    if item.field not in ['num', 'QA', 'Category', 'Filename']:
        raise HTTPException(status_code=400, detail="Invalid field")

    if item.field == 'num':
        item.value = int(item.value)
    filtered_data = data[data[item.field] == item.value]
    return JSONResponse(content=filtered_data.to_dict(orient='records'))

@app.delete("/delete_csv_data")
async def delete_csv_data(item: SearchItem):
    data = pd.read_csv('Database.csv', usecols=['num', 'QA', 'Category', 'Filename'])
    if item.field not in ['num', 'QA', 'Category', 'Filename']:
        raise HTTPException(status_code=400, detail="Invalid field")

    if item.field == 'num':
        item.value = int(item.value)

    filtered_data = data[data[item.field] != item.value]
    
    filtered_data.to_csv('Database.csv', index=False)
    
    return JSONResponse(content={"message": "Data deleted successfully"})
