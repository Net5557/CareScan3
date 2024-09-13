from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import io
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files (เช่น รูปภาพ, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# โหลดโมเดล YOLO
model = YOLO("last.pt")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open('templates/index.html', 'r', encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/data.html", response_class=HTMLResponse)
async def data():
    with open('templates/data.html', 'r', encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/contact.html", response_class=HTMLResponse)
async def data():
    with open('templates/contact.html', 'r', encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/map.html", response_class=HTMLResponse)
async def data():
    with open('templates/map.html', 'r', encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    np_image = np.frombuffer(image_data, np.uint8)
    np_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    results = model.predict(source=np_image)
    result = results[0]

    predictions = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        predictions.append({
            "class_id": class_id,
            "coordinates": cords,
            "confidence": conf
        })

    result_image = Image.fromarray(result.plot()[:,:,::-1])
    buffer = io.BytesIO()
    result_image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Encode image as base64
    result_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return JSONResponse(content={"predictions": predictions, "result_image": result_image_base64})
