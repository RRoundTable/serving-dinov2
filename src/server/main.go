// Package main contains the apis served by echo.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"      // Package http provides HTTP client and server implementations.
	_ "server/docs" // docs is generated by Swag CLI, you have to import it.

	triton "server/tritonserver"

	"github.com/labstack/echo-contrib/pprof"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	echoSwagger "github.com/swaggo/echo-swagger"
	_ "go.uber.org/automaxprocs"
)

// Flags contains the information to send requests to Triton inference server.
type Flags struct {
	URL     string
	PORT    string
	TIMEOUT int64
	PROFILE bool
}

// parseFlags parses the arguments and initialize the flags.
func parseFlags() Flags {
	var flags = Flags{}
	// https://github.com/NVIDIA/triton-inference-server/tree/master/docs/examples/model_repository/simple
	flag.StringVar(
		&flags.URL,
		"u",
		"localhost:8001",
		"Inference Server URL. Default: localhost:8001",
	)
	flag.StringVar(&flags.PORT, "p", "20000", "Service Port. Default: 20000")
	flag.Int64Var(&flags.TIMEOUT, "t", 10, "Timeout. Default: 10 Sec.")
	flag.BoolVar(&flags.PROFILE, "profile", false, "Enable profliling.")
	flag.Parse()
	return flags
}

// @title         Inference API Server
// @description   profiling - http://localhost:20000/debug/pprof/
// @contact.name  Team ML
// @contact.email TeamML@annotation-ai.com.
func main() {
	// Parse the args.
	flags := parseFlags()
	log.Println("Flags:", flags)

	// Check the gRPC connection well-established.
	client := triton.NewGRPCInferenceServiceAPIClient(flags.URL, flags.TIMEOUT)

	// Create a server with echo.
	e := echo.New()
	// Logger middleware logs the information about each HTTP request.
	e.Use(middleware.Logger())
	if flags.PROFILE {
		pprof.Register(e)
		log.Println("Profiler On")
	}

	// APIs.
	e.GET("/", getHealthCheck)
	e.GET("/liveness", client.GetServerLiveness)
	e.GET("/readiness", client.GetServerReadiness)
	e.GET("/model-metadata", client.GetModelMetadata)
	e.GET("/model-stats", client.GetModelInferStats)
	e.POST("/model-load", client.LoadModel)
	e.POST("/model-unload", client.UnloadModel)
	e.POST("/infer", client.Infer)

	// Swagger.
	e.GET("/docs/*", echoSwagger.WrapHandler)
	e.Logger.Fatal(e.Start(":" + flags.PORT))
}

// @Summary     Healthcheck
// @Description It returns true if the api server is alive.
// @Accept      json
// @Produce     json
// @Success     200 {object} bool "API server's liveness"
// @Router      / [get].
func getHealthCheck(c echo.Context) error {
	return fmt.Errorf("%w", c.JSON(http.StatusOK, true))
}
