import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI
# import tf
import tensorflow as tf

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Intent API!"}


@app.post("/predict")
async def get_intent_prediction(message: str = ""):
    if message == "":
        return {"message": "No message provided"}
    
    
    intent_model = tf.keras.models.load_model('saved_model/my_model')
    print(intent_model)
    file = open("labels.pkl",'rb')
    binarizer = pickle.load(file)
    print(binarizer)

    results = tf.nn.softmax(intent_model(tf.constant([message])))
    intents = binarizer.inverse_transform(results.numpy())



    return {
        "message": message,
        "model-prediction": intents[0]
    }


ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)