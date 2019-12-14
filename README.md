# Hashtag Prediction with Pytorch

Multimodal hashtag prediction from instagram

## Overview

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/70353952-d6307780-18b1-11ea-9db4-f38399a48dc0.png" />  
</p>

## Model Architecture

- `ALBERT` for text, `VGG16` for image
- Take out the `[CLS]` token from ALBERT, and change it to 100 dim
- Change to 100 dim after flattening the VGG16 output
- Concat them, and predict among 100 labels.

## Dataset

- Collect 50000 data from instagram (w/ selenium crawler)
- Only include English data

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
# URL
localhost:80/predict?image_id=1DGu9R5a9jpkY-fy79VrGFmCdJigzTMC-&text=20%20days%20till%20Christmas%20%F0%9F%98%8D%F0%9F%8E%85&max_seq_len=20&n_label=10
```

## Run on Ainize

[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=github.com/monologg/hashtag-prediction-pytorch)

<p float="left" align="left">
    <img width="400" src="https://user-images.githubusercontent.com/28896432/70370210-90060300-1907-11ea-882c-f7c2251971f5.png" />

1. `image_id` : the share id you can get from google drive above
2. `text` : like caption in instagram
3. `max_seq_len`: maximum sequence length
4. `n_label`: num of labels you want to predict

```bash
https://endpoint.ainize.ai/monologg/hashtag/predict?image_id={image_id}&text={text}&max_seq_len={max_seq_len}&n_label={n_label}
```

```bash
# URL
https://endpoint.ainize.ai/monologg/hashtag/predict?image_id=1DGu9R5a9jpkY-fy79VrGFmCdJigzTMC-&text=20%20days%20till%20Christmas%20%F0%9F%98%8D%F0%9F%8E%85&max_seq_len=20&n_label=10
```

### Result on html

<p float="left" align="left">
    <img width="400" src="https://user-images.githubusercontent.com/28896432/70370717-f857e300-190d-11ea-8804-b9ca6b481f85.png" />

## Reference

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [ALBERT Paper](https://arxiv.org/abs/1909.11942)
