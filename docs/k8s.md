# Kubernetes

- Learn how to deploy deeplearning model dinov2 on k8s with practical examples.
- Includes everything from Model Serving to Monitoring.

## Diagram

![image](https://github.com/RRoundTable/serving-dinov2/assets/27891090/02d1e30b-b169-4d2e-bba0-8c9481397007)

## Prerequisites
- Ubuntu 20.04
- Nvidia GPU
- Nvidia Container Toolkit

## Setup

```
make cluster
```

Check dcgm-exporter.

```
kubectl get pod
```

```
NAME                  READY   STATUS    RESTARTS   AGE
dcgm-exporter-zns7m   1/1     Running   0          4d3h
```


## Deploy charts

```
make charts
```

```
kubectl get pod
```

```
NAME                                                     READY   STATUS    RESTARTS   AGE
dcgm-exporter-zns7m                                      1/1     Running   0          4d3h
minio-69966fb668-ltzp9                                   1/1     Running   0          45s
traefik-677c7d64f8-mdk8s                                 1/1     Running   0          45s
loki-0                                                   0/1     Running   0          34s
prometheus-kube-prometheus-operator-7958587c67-8fccb     1/1     Running   0          24s
prometheus-prometheus-node-exporter-89fkv                1/1     Running   0          24s
alertmanager-prometheus-kube-prometheus-alertmanager-0   2/2     Running   0          23s
triton-prometheus-adapter-674d9855f-8jkqm                0/1     Running   0          18s
client-serving-dinov2-client-68cfdf5d9c-n842k            1/1     Running   0          17s
prometheus-grafana-9bcd4c849-z44t6                       3/3     Running   0          24s
prometheus-kube-state-metrics-85c858f4b-9bbj5            1/1     Running   0          24s
promtail-4jr62                                           1/1     Running   0          33s
triton-55df95678f-b9g2r                                  1/1     Running   0          18s
prometheus-prometheus-kube-prometheus-prometheus-0       2/2     Running   0          23s
client-serving-dinov2-client-68cfdf5d9c-xvk4q            1/1     Running   0          2s
```

## Client

You can run gradio demo about dinov2 pca.

Access http://localhost:7860


## Grafana

Access http://localhost:3000

ID: admin
PW: prom-operator

You can check prometheus and loki.

## Traefik

We use traefik as a grpc loadbalancer.

## Remove charts

```
make remove-charts
```

## Finalize

Remove cluster

```
make finalize
```