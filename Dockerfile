FROM golang:1.25-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY atlas ./atlas
RUN CGO_ENABLED=0 go build -o /atlas ./cmd/atlas

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /atlas /atlas
ENTRYPOINT ["/atlas"]
