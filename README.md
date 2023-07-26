# serving-dinov2

## Contents

- [x] Serving dinov2 onnx model with triton. ([Artifact](https://huggingface.co/RoundtTble/dinov2_vitl14_trt_a4000))
- [x] Gradio Demo.
- [x] Docker Compose.
- [x] K8s Setting(Triton, Traefik, Promtail, Loki, Prometheus, Grafana).
- [ ] Serving dinov2 onnx model with TensorRT.
- [ ] ~Serving dinov2 onnx model with Fastertransformer~([fastertransformer_backend](https://github.com/triton-inference-server/fastertransformer_backend) don't support vit yet.)

## Docker Compose

Check [docker-compose](docs/docker-compose.md)

## Kubernetes

![image](https://github.com/RRoundTable/serving-dinov2/assets/27891090/02d1e30b-b169-4d2e-bba0-8c9481397007)
Check [Kubernetes](docs/k8s.md)