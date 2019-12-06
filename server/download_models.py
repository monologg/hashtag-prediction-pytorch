from tensorflow.keras.applications.vgg16 import VGG16
from app import download_model
from transformers import AlbertModel

vgg_model = VGG16(weights='imagenet', include_top=False)
download_model()

model = AlbertModel.from_pretrained('albert-base-v2')
del(model)
