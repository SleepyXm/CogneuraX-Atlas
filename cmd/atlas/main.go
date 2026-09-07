package main

import (
	"context"
	"database/sql"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/SleepyXm/CogneuraX-Atlas/atlas"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/joho/godotenv"
)

func main() {
	_ = godotenv.Load()
	cfg, err := atlas.LoadConfig()
	if err != nil {
		log.Fatal(err)
	}
	db, err := sql.Open("pgx", cfg.DatabaseURL)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	if err := db.Ping(); err != nil {
		log.Fatalf("database is not reachable: %v", err)
	}

	service, err := atlas.NewService(cfg, db)
	if err != nil {
		log.Fatal(err)
	}
	defer service.Close()
	prepare, cancelPrepare := context.WithTimeout(context.Background(), 30*time.Second)
	err = service.PrepareInfrastructure(prepare)
	cancelPrepare()
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	workerDone := make(chan error, 1)
	go func() { workerDone <- service.RunWorker(ctx) }()
	server := &http.Server{Addr: cfg.Addr, Handler: service.Router(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	log.Printf("CogneuraX Atlas listening on %s", cfg.Addr)
	serveErr := server.ListenAndServe()
	stop()
	if workerErr := <-workerDone; workerErr != nil {
		log.Printf("Atlas worker stopped: %v", workerErr)
	}
	if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
		log.Fatal(serveErr)
	}
}
