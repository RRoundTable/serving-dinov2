# Docker Compose

## Prerequisites

- [docker compose](https://docs.docker.com/compose/install/)
- Ubuntu 20.04
- Nvidia GPU

## Setup

Clone dinov2 model from huggingface.

```
make model
```

Check `model_repository`.

```
tree model_repository
```

```
model_repository
└── dinov2_vitl14
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

## Run

```
docker compose up
```

Access http://localhost:7860


## Finalize

```
docker compose down
```