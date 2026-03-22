.PHONY: build

CXX = g++

CXXFLAGS = $(shell pkg-config --cflags opencv4) \
			-I/usr/local/include/onnxruntime \
			-O2

LDFLAGS = $(shell pkg-config --libs opencv4) \
			-L/usr/local/lib \
			-lonnxruntime
			
TARGET = detect

all: $(TARGET)

$(TARGET): detect.c coco_labels.h
	$(CXX) -o $(TARGET) detect.c $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

# Docker stuff
reattach:
	docker compose down
	docker compose up -d
	docker exec -it gstreamer_custom bash

attach:
	docker exec -it gstreamer_custom bash

build:
	docker compose up --build -d
