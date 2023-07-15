PROFILE_NAME=serving-dinov2


model:
	git lfs install
	git clone git@hf.co:RoundtTble/dinov2_vitl14_onnx
	cp -r dinov2_vitl14_onnx/model_repository ./model_repository
	rm -rf dinov2_vitl14_onnx

minikube:
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo install minikube-linux-amd64 /usr/local/bin/minikube


cluster:
	minikube start driver=docker --profile=$(PROFILE_NAME)
	eval $(minikube docker-env) # for using host docker images
