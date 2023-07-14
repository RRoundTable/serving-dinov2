
model:
	git lfs install
	git clone git@hf.co:RoundtTble/dinov2_vitl14_onnx
	cp -r dinov2_vitl14_onnx/model_repository ./model_repository
	rm -rf dinov2_vitl14_onnx

