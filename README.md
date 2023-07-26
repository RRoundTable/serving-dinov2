# serving-dinov2

## Contents

- [x] Serving dinov2 onnx model with triton. ([Artifact](https://huggingface.co/RoundtTble/dinov2_onnx) `throughput: 48.927 infer/sec`)
- [x] Gradio Demo.
- [x] Docker Compose.
- [x] K8s Setting(Triton, Traefik, Promtail, Loki, Prometheus, Grafana).
- [x] Serving dinov2 TensorRT Model. ([Artifact](https://huggingface.co/RoundtTble/dinov2_trt_a4000) `throughput: 222.66 infer/sec`)
- [ ] ~~Serving dinov2 onnx model with Fastertransformer~~ ([fastertransformer_backend](https://github.com/triton-inference-server/fastertransformer_backend) don't support vit yet.)

## Docker Compose

Check [docker-compose](docs/docker-compose.md)

## Kubernetes

![image](https://github.com/RRoundTable/serving-dinov2/assets/27891090/02d1e30b-b169-4d2e-bba0-8c9481397007)
Check [Kubernetes](docs/k8s.md)