from tensorflow.keras.applications.vgg16 import VGG16
from app import download_model

vgg_model = VGG16(weights='imagenet', include_top=False)
download_model()
