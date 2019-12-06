# Hashtag Prediction with Pytorch

Multimodal hashtag prediction from instagram

## Overview

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/70353952-d6307780-18b1-11ea-9db4-f38399a48dc0.png" />  
</p>

## Model Architecture

- `ALBERT` for text, `VGG16` for image
- Take out the `[CLS]` token from ALBERT, and change it to 100 dim
- Change 100 dim after flattening the VGG16 output
- Concat them, and predict among 100 labels.

## How to use

### 1. Run docker server

```bash
$ docker run -d -p 80:80 adieujw/hashtag:latest
```

### 2. Put image at Google Drive

- Put your image in [this google drive](https://drive.google.com/drive/folders/1m0lkcMIajII8aaqQHsTlji4dLjo9X_1_)

<p float="left" align="left">
    <img width="400" src="https://user-images.githubusercontent.com/28896432/70354672-92d70880-18b3-11ea-91f7-65a75a8ed8ea.png" />

- Copy the id. It will be used when you give request.

  <p float="left" align="left">
      <img width="300" src="https://user-images.githubusercontent.com/28896432/70354636-71761c80-18b3-11ea-854c-ee2137f3e8b5.png" />

### 3. Request

```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"image_id":"1oKJeos4q19l07o82UhcDqDKPxdULX38q","text":"I am very cool.", "max_seq_len":20,"n_label":10}' http://0.0.0.0:80/predict
```

## Reference

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [ALBERT Paper]()
