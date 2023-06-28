package tritonserver

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http" // Package http provides HTTP client and server implementations.
	"time"

	"github.com/labstack/echo/v4"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Contexts that is necessary to communicate with Triton.
type GRPCInferenceServiceAPIs interface {
	GetServerLiveness(c echo.Context) error
	GetServerReadiness(c echo.Context) error
	GetModelMetadata(c echo.Context) error
	GetModelInferStats(c echo.Context) error
	LoadModel(c echo.Context) error
	UnloadModel(c echo.Context) error
	Infer(c echo.Context) error
}

// Contexts that is necessary to communicate with Triton.
type GRPCInferenceServiceAPIClient struct {
	grpc    GRPCInferenceServiceClient
	url     string
	timeout int64
}

// The inference response w/ a single input and a single output.
type InferResponse struct {
	Output string  `json:"output" xml:"output"`
	Dtype  string  `json:"dtype"  xml:"dtype"`
	Shape  []int64 `json:"shape"  xml:"shape"`
}

// ConnectToTritonWithGRPC Create GRPC Connection.
func NewGRPCInferenceServiceAPIClient(url string, timeout int64) GRPCInferenceServiceAPIClient {
	conn, err := grpc.Dial(url, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("couldn't connect to endpoint %s: %v", url, err)
	}
	client := GRPCInferenceServiceAPIClient{}
	client.grpc = NewGRPCInferenceServiceClient(conn)
	client.url = url
	client.timeout = timeout
	return client
}

// Convert Dtype of the model inference response.
func convertDtypeModelInferResponse(dtype string) (string, error) {
	var err error
	switch dtype {
	case "BOOL":
		dtype, err = "bool", nil
	case "INT8":
		dtype, err = "INT8", nil
	case "INT16":
		dtype, err = "int16", nil
	case "INT32":
		dtype, err = "int32", nil
	case "INT64":
		dtype, err = "int64", nil
	case "UINT8":
		dtype, err = "uint8", nil
	case "UINT16":
		dtype, err = "uint16", nil
	case "UINT32":
		dtype, err = "uint32", nil
	case "UINT64":
		dtype, err = "uint64", nil
	case "FP16":
		dtype, err = "float16", nil
	case "FP32":
		dtype, err = "float32", nil
	case "FP64":
		dtype, err = "float64", nil
	default:
		err = fmt.Errorf("couldn't convert type %s", dtype)
	}
	return dtype, err
}

// @Summary     Check Triton's liveness.
// @Description It returns true if the triton server is alive.
// @Accept      json
// @Produce     json
// @Success     200 {object} bool "Triton server's liveness"
// @Router      /liveness [get].
func (client *GRPCInferenceServiceAPIClient) GetServerLiveness(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	serverLiveRequest := ServerLiveRequest{}
	serverLiveResponse, err := client.grpc.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		return fmt.Errorf("failed to get server live response: %w", err)
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, serverLiveResponse.Live))
}

// @Summary     Check Triton's Readiness.
// @Description It returns true if the triton server is ready.
// @Accept      json
// @Produce     json
// @Success     200 {object} bool "Triton server's readiness"
// @Router      /readiness [get].
func (client *GRPCInferenceServiceAPIClient) GetServerReadiness(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	serverReadyRequest := ServerReadyRequest{}
	serverReadyResponse, err := client.grpc.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		return fmt.Errorf("failed to get server ready response: %w", err)
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, serverReadyResponse.Ready))
}

// @Summary     Get model metadata.
// @Description It returns the requested model metadata
// @Accept      json
// @Produce     json
// @Param       model_name    query    string                true  "model name"
// @Param       model_version query    string                false "model version"
// @Success     200           {object} ModelMetadataResponse "Triton server's model metadata"
// @Router      /model-metadata [get].
func (client *GRPCInferenceServiceAPIClient) GetModelMetadata(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	modelMetadataRequest := ModelMetadataRequest{
		Name:    c.QueryParam("model_name"),
		Version: c.QueryParam("model_version"),
	}
	modelMetadataResponse, err := client.grpc.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		return fmt.Errorf("failed to get model metadata response: %w", err)
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, modelMetadataResponse))
}

// @Summary     Get model inference statistics.
// @Description It returns the requested model's inference statistics.
// @Accept      json
// @Produce     json
// @Param       model_name    query    string                  true  "model name"
// @Param       model_version query    string                  false "model version"
// @Success     200           {object} ModelStatisticsResponse "Triton server's model statistics"
// @Router      /model-stats [get].
func (client *GRPCInferenceServiceAPIClient) GetModelInferStats(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	modelStatisticsRequest := ModelStatisticsRequest{
		Name:    c.QueryParam("model_name"),
		Version: c.QueryParam("model_version"),
	}
	modelStatisticsResponse, err := client.grpc.ModelStatistics(ctx, &modelStatisticsRequest)
	if err != nil {
		return fmt.Errorf("failed to get model statistics response: %w", err)
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, modelStatisticsResponse))
}

// @Summary     Load a model.
// @Description It requests to load a model. This is only allowed when polling is enabled.
// @Accept      json
// @Produce     json
// @Param       model_name query    string                      true "model name"
// @Success     200        {object} RepositoryModelLoadResponse "Triton server's model load response"
// @Router      /model-load [post].
func (client *GRPCInferenceServiceAPIClient) LoadModel(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	modelLoadRequest := RepositoryModelLoadRequest{ModelName: c.QueryParam("model_name")}
	modelLoadResponse, err := client.grpc.RepositoryModelLoad(ctx, &modelLoadRequest)
	if err != nil {
		return fmt.Errorf("failed to get model load response: %w", err)
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, modelLoadResponse))
}

// @Summary     Unload a model.
// @Description It requests to unload a model. This is only allowed when polling is enabled.
// @Accept      json
// @Produce     json
// @Param       model_name query    string                        true "model name"
// @Success     200        {object} RepositoryModelUnloadResponse "Triton server's model unload response"
// @Router      /model-unload [post].
func (client *GRPCInferenceServiceAPIClient) UnloadModel(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	modelUnloadRequest := RepositoryModelUnloadRequest{ModelName: c.QueryParam("model_name")}
	modelUnloadResponse, err := client.grpc.RepositoryModelUnload(ctx, &modelUnloadRequest)
	if err != nil {
		return fmt.Errorf("failed to get model unload response: %w", err)
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, modelUnloadResponse))
}

// @Summary     Model inference api for the model with bytes a input and a bytes output.
// @Description It outputs a single bytes with a single bytes input.
// @Accept      json
// @Produce     json
// @Param       modelName    query    string             true  "model name"
// @Param       modelVersion query    string             false "model version"
// @Param       file         formData file               true  "input"
// @Success     200          {object} ModelInferResponse "Triton server's inference response"
// @Router      /infer [post].
func (client *GRPCInferenceServiceAPIClient) Infer(c echo.Context) error {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		time.Duration(client.timeout)*time.Second,
	)
	defer cancel()

	// Get the model information.
	modelName := c.QueryParam("modelName")
	modelVersion := c.QueryParam("modelVersion")

	// Get the file.
	file, err := c.FormFile("file")
	if err != nil {
		return fmt.Errorf("failed to get form file: %w", err)
	}
	fileContent, err := file.Open()
	if err != nil {
		return fmt.Errorf("failed to get file content from form file: %w", err)
	}
	defer fileContent.Close()
	rawInput, err := io.ReadAll(fileContent)
	if err != nil {
		return fmt.Errorf("failed to get raw input from file content: %w", err)
	}

	// Create request input / output tensors.
	size := int64(len(rawInput))
	inferInputs := []*ModelInferRequest_InferInputTensor{
		{Name: "INPUT", Datatype: "UINT8", Shape: []int64{size}},
	}
	inferOutputs := []*ModelInferRequest_InferRequestedOutputTensor{{Name: "OUTPUT"}}

	// Create a request.
	modelInferRequest := ModelInferRequest{
		ModelName:        modelName,
		ModelVersion:     modelVersion,
		Inputs:           inferInputs,
		Outputs:          inferOutputs,
		RawInputContents: [][]byte{rawInput},
	}

	// Get infer response.
	modelInferResponse, err := client.grpc.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		return fmt.Errorf("failed to get model infer response: %w", err)
	}

	dtype, err := convertDtypeModelInferResponse(modelInferResponse.Outputs[0].Datatype)
	if err != nil {
		return fmt.Errorf("%w", err)
	}
	inferResponse := InferResponse{
		Output: base64.StdEncoding.EncodeToString(modelInferResponse.RawOutputContents[0]),
		Dtype:  dtype,
		Shape:  modelInferResponse.Outputs[0].Shape,
	}
	return fmt.Errorf("%w", c.JSON(http.StatusOK, inferResponse))
}
