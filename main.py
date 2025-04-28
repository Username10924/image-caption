from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM
)
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io

app = FastAPI()

# --- List of available models ---
models = {
    "1": {
        "name": "BLIP Base",
        "processor": "Salesforce/blip-image-captioning-base",
        "model": "Salesforce/blip-image-captioning-base",
        "type": "blip"
    },
    "2": {
        "name": "BLIP Large",
        "processor": "Salesforce/blip-image-captioning-large",
        "model": "Salesforce/blip-image-captioning-large",
        "type": "blip"
    },
    "3": {
        "name": "BLIP-2 OPT 2.7B",
        "processor": "Salesforce/blip2-opt-2.7b",
        "model": "Salesforce/blip2-opt-2.7b",
        "type": "blip2"
    },
    "4": {
        "name": "GIT Base",
        "processor": "microsoft/git-base",
        "model": "microsoft/git-base",
        "type": "git"
    }
}

# to avoid loading models again every request
loaded_models = {}

def get_model(choice):
    if choice not in models:
        raise HTTPException(status_code=400, detail="invalid model picked")
    if choice in loaded_models:
        return loaded_models[choice]
    selected = models[choice]
    # load processor and model based on choice
    if selected["type"] == "blip":
        processor = BlipProcessor.from_pretrained(selected["processor"])
        model = BlipForConditionalGeneration.from_pretrained(selected["model"])
    elif selected["type"] == "blip2":
        processor = Blip2Processor.from_pretrained(selected["processor"])
        model = Blip2ForConditionalGeneration.from_pretrained(selected["model"])
    elif selected["type"] == "git":
        processor = AutoProcessor.from_pretrained(selected["processor"])
        model = AutoModelForCausalLM.from_pretrained(selected["model"])
    else:
        raise HTTPException(status_code=400, detail="invalid model type")
    # cache the loaded model
    loaded_models[choice] = (selected["type"], processor, model)
    return selected["type"], processor, model


@app.post("/caption")
async def generate_caption(
    model_number: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        image_bytes = await image.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")
    
    # get model type, processor and model
    model_type, processor, model = get_model(model_number)
    if model_type in ["blip", "blip2"]:
        inputs = processor(raw_image, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
    elif model_type =="git":
        pixel_values = processor(images=raw_image, return_tensors="pt")
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        raise HTTPException(status_code=400, detail="invalid model type")
    
    return JSONResponse({"caption": caption})
        
