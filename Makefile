# Makefile: Common ML Commands

# Training
train-seg:
	python scripts/train_seg.py --config configs/seg.yaml

train-ssl:
	python scripts/train_ssl.py --config configs/ssl.yaml

# Evaluation
eval:
	python scripts/evaluate.py --config configs/seg.yaml

# Inference
infer:
	python scripts/infer.py --input data/tiles/test --output outputs/

# DVC
dvc-push:
	dvc push

dvc-pull:
	dvc pull

# Docker
build-docker-train:
	docker build -f docker/Dockerfile.train -t footprint-train .

build-docker-infer:
	docker build -f docker/Dockerfile.infer -t footprint-infer .

# FastAPI
serve-api:
	uvicorn api.main:app --reload --port 8000