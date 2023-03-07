
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import cv2 as cv


from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from PIL import Image
from io import BytesIO



project_path="C:/Users/pc/Nextcloud/Python/GITHUB/FastApi-app/"
static_dir=project_path+"index_files/"
model_path=project_path+'models/'


app=FastAPI()

#app.mount("/index_files", StaticFiles(directory=static_dir), name="static")

templates=Jinja2Templates(directory=project_path)



@app.get('/', response_class=HTMLResponse)
def get_basic_form(request: Request):
    return templates.TemplateResponse('index.html',{"request":request})
    
@app.post('/', response_class=HTMLResponse)
async def post_function(request:Request, file:UploadFile=File(...)):
    
    image=await file.read()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        rst= "Image must be jpg or png format!"
        
    #load saved model
    model_loaded=keras.models.load_model(model_path)
    with open(f"{project_path}{file.filename}","wb") as f:
        f.write(image)  
        
    #load image
    image=cv.imread(project_path+file.filename)
    #plt.imshow(image)
    
    #convert image to gray and resize
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,[28,28])
    data=np.reshape(image, [-1,28,28,1])
    
        # predict 
    pred=model_loaded.predict(data)
    rst=str(np.argmax(pred))
    
    return templates.TemplateResponse('index.html', {"request" : request, "Results": rst})
    
    
    
if __name__ == "__main__":
    uvicorn.run(app)