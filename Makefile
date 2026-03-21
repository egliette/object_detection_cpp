.PHONY: build

CC = gcc
CFLAGS = $(shell pkg-config --cflags gstreamer-1.0)
LIBS = $(shell pkg-config --libs gstreamer-1.0)

TARGET = pipeline
SRC = pipeline.c 

# Compile
compile:
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LIBS)

# Docker stuff
reattach:
	docker compose down
	docker compose up -d
	docker exec -it gstreamer_custom bash

attach:
	docker exec -it gstreamer_custom bash

build:
	docker compose up --build -d
