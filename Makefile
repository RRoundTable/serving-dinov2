
model:
	git lfs install
	git clone git@hf.co:RoundtTble/dinov2_vitl14_onnx
	cp -r dinov2_vitl14_onnx/model_repository ./model_repository
	rm -rf dinov2_vitl14_onnx

cluster:
	curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="v1.27.2+k3s1" K3S_KUBECONFIG_MODE="644" INSTALL_K3S_EXEC="server --disable=traefik" sh -s - --docker
	mkdir -p ~/.kube
	cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
	kubectl create -f https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/master/dcgm-exporter.yaml
	helm repo add traefik https://helm.traefik.io/traefik
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update

finalize:
	sh /usr/local/bin/k3s-killall.sh
	sh /usr/local/bin/k3s-uninstall.sh

.PHONY: charts
charts:
	helm install traefik charts/traefik
	helm install loki charts/loki
	helm install promtail charts/promtail
	helm install prometheus charts/prometheus
	helm install triton charts/triton
	helm install client charts/client

remove-charts:
	helm uninstall client || true
	helm uninstall triton || true
	helm uninstall prometheus || true
	helm uninstall promtail || true
	helm uninstall loki || true
	helm uninstall traefik || true
