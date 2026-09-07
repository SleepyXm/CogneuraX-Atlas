package atlas

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

func newObjectStore() (objectStore, error) {
	switch env("ATLAS_STORAGE", "local") {
	case "local":
		root, err := filepath.Abs(env("ATLAS_DATA_DIR", "data/documents"))
		if err != nil {
			return nil, fmt.Errorf("resolve Atlas data directory: %w", err)
		}
		if err := os.MkdirAll(root, 0o750); err != nil {
			return nil, fmt.Errorf("create Atlas data directory: %w", err)
		}
		return &localStore{root: root}, nil
	case "s3":
		endpoint, accessKey, secretKey, bucket := os.Getenv("S3_ENDPOINT"), os.Getenv("S3_ACCESS_KEY"), os.Getenv("S3_SECRET_KEY"), os.Getenv("S3_BUCKET")
		if endpoint == "" || accessKey == "" || secretKey == "" || bucket == "" {
			return nil, errors.New("S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, and S3_BUCKET are required for s3 storage")
		}
		secure, err := boolEnv("S3_USE_TLS", true)
		if err != nil {
			return nil, err
		}
		client, err := minio.New(endpoint, &minio.Options{
			Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
			Secure: secure,
		})
		if err != nil {
			return nil, fmt.Errorf("configure S3 storage: %w", err)
		}
		return &s3Store{client: client, bucket: bucket}, nil
	default:
		return nil, fmt.Errorf("unsupported ATLAS_STORAGE %q", os.Getenv("ATLAS_STORAGE"))
	}
}

func (s *localStore) path(key string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(key))
	if clean == "." || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("invalid storage key")
	}
	path := filepath.Join(s.root, clean)
	if path != s.root && !strings.HasPrefix(path, s.root+string(filepath.Separator)) {
		return "", errors.New("storage key escapes configured root")
	}
	return path, nil
}

func (s *localStore) Put(ctx context.Context, key string, src io.Reader, maxBytes int64) (storedObject, error) {
	path, err := s.path(key)
	if err != nil {
		return storedObject{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return storedObject{}, err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return storedObject{}, err
	}
	hash := sha256.New()
	size, copyErr := io.Copy(io.MultiWriter(file, hash), io.LimitReader(src, maxBytes+1))
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil || size > maxBytes || ctx.Err() != nil {
		_ = os.Remove(path)
		if copyErr != nil {
			return storedObject{}, copyErr
		}
		if closeErr != nil {
			return storedObject{}, closeErr
		}
		if ctx.Err() != nil {
			return storedObject{}, ctx.Err()
		}
		return storedObject{}, clientError{status: 413, message: "document exceeds the upload limit"}
	}
	return storedObject{SHA256: hex.EncodeToString(hash.Sum(nil)), Size: size}, nil
}

func (s *localStore) Open(_ context.Context, key string) (io.ReadCloser, error) {
	path, err := s.path(key)
	if err != nil {
		return nil, err
	}
	return os.Open(path)
}

func (s *localStore) Delete(_ context.Context, key string) error {
	path, err := s.path(key)
	if err != nil {
		return err
	}
	err = os.Remove(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func (s *s3Store) Put(ctx context.Context, key string, src io.Reader, maxBytes int64) (storedObject, error) {
	hash := sha256.New()
	// Unknown-length multipart upload keeps the original streaming; the reader
	// limits and hashes the same bytes sent to S3.
	info, err := s.client.PutObject(ctx, s.bucket, key, io.TeeReader(io.LimitReader(src, maxBytes+1), hash), -1, minio.PutObjectOptions{})
	if err != nil {
		return storedObject{}, err
	}
	if info.Size > maxBytes {
		if err := s.Delete(ctx, key); err != nil {
			return storedObject{}, fmt.Errorf("document exceeds upload limit and S3 cleanup failed: %w", err)
		}
		return storedObject{}, clientError{status: 413, message: "document exceeds the upload limit"}
	}
	return storedObject{SHA256: hex.EncodeToString(hash.Sum(nil)), Size: info.Size}, nil
}

func (s *s3Store) Open(ctx context.Context, key string) (io.ReadCloser, error) {
	return s.client.GetObject(ctx, s.bucket, key, minio.GetObjectOptions{})
}

func (s *s3Store) Delete(ctx context.Context, key string) error {
	return s.client.RemoveObject(ctx, s.bucket, key, minio.RemoveObjectOptions{})
}
