
model:
	git lfs install
	git clone git@hf.co:RoundtTble/dinov2_vits14_onnx
	cp -r dinov2_vits14_onnx/model_repository ./model_repository
	rm -rf dinov2_vits14_onnx

