#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{"image_id":"1oKJeos4q19l07o82UhcDqDKPxdULX38q","text":"I am very cool", "max_seq_len":20,"n_label":10}' http://0.0.0.0:80/predict
