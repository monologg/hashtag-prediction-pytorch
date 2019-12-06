#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{"image_id":"1oKJeos4q19l07o82UhcDqDKPxdULX38q","text":"I love this one", "max_seq_len":20}' http://0.0.0.0:80/predict
