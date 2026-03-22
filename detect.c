#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>
#include "coco_labels.h"

#define MODEL_INPUT_W   640
#define MODEL_INPUT_H   640
#define NUM_PREDICTIONS 8400
#define NUM_OUTPUTS     84

#define ORT_CHECK(expr)                                              \
    do {                                                             \
        OrtStatus *_s = (expr);                                      \
        if (_s != NULL) {                                            \
            fprintf(stderr, "ONNX Runtime error at line %d: %s\n",  \
                    __LINE__, ort->GetErrorMessage(_s));             \
            ort->ReleaseStatus(_s);                                  \
            goto cleanup;                                            \
        }                                                            \
    } while (0)

/* -----------------------------------------------------------------------
 * IoU — intersection over union of two boxes [cx, cy, w, h]
 * ----------------------------------------------------------------------- */
static float iou(float *a, float *b)
{
    float ax1 = a[0] - a[2]/2, ay1 = a[1] - a[3]/2;
    float ax2 = a[0] + a[2]/2, ay2 = a[1] + a[3]/2;
    float bx1 = b[0] - b[2]/2, by1 = b[1] - b[3]/2;
    float bx2 = b[0] + b[2]/2, by2 = b[1] + b[3]/2;

    float ix1 = ax1 > bx1 ? ax1 : bx1;
    float iy1 = ay1 > by1 ? ay1 : by1;
    float ix2 = ax2 < bx2 ? ax2 : bx2;
    float iy2 = ay2 < by2 ? ay2 : by2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.0f;

    float inter_area = iw * ih;
    float a_area     = a[2] * a[3];
    float b_area     = b[2] * b[3];
    float union_area = a_area + b_area - inter_area;

    return inter_area / union_area;
}

typedef struct {
    float x, y, w, h;
    float score;
    int   class_id;
    int   keep;
} Detection;

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* --- declare ALL variables at the top so goto cleanup is always safe --- */
    OrtStatus         *status      = NULL;
    OrtEnv            *env         = NULL;
    OrtSessionOptions *opts        = NULL;
    OrtSession        *session     = NULL;
    OrtAllocator      *allocator   = NULL;
    OrtMemoryInfo     *mem_info    = NULL;
    char              *input_name  = NULL;
    char              *output_name = NULL;
    float             *input_data  = NULL;
    int64_t            input_dims[4];
    size_t             input_size;

    /* cv::Mat declared here — default constructor, no initialization needed */
    cv::Mat frame;
    cv::Mat resized;
    int frame_idx = 0;

    /* --- argument parsing --- */
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.mp4> <model.onnx> <output.mp4> [conf_thresh]\n", argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *model_path  = argv[2];
    const char *output_path = argv[3];
    float conf_thresh = (argc >= 5) ? (float)atof(argv[4]) : 0.45f;
    float nms_thresh  = 0.45f;

    printf("Input:  %s\n", input_path);
    printf("Model:  %s\n", model_path);
    printf("Output: %s\n", output_path);
    printf("Conf threshold: %.2f\n", conf_thresh);

    /* --- open input video --- */
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        fprintf(stderr, "Error: cannot open input video: %s\n", input_path);
        return 1;
    }

    int    frame_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int    frame_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps     = cap.get(cv::CAP_PROP_FPS);
    int    total   = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    printf("Video: %dx%d @ %.1f fps, %d frames\n", frame_w, frame_h, fps, total);

    /* --- open output video writer --- */
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(output_path, fourcc, fps, cv::Size(frame_w, frame_h));
    if (!writer.isOpened()) {
        fprintf(stderr, "Error: cannot open output video: %s\n", output_path);
        return 1;
    }

    /* --- initialize ONNX Runtime --- */
    const OrtApi *ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    ORT_CHECK(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "detect", &env));
    ORT_CHECK(ort->CreateSessionOptions(&opts));
    ORT_CHECK(ort->CreateSession(env, model_path, opts, &session));
    printf("Model loaded.\n");

    ORT_CHECK(ort->GetAllocatorWithDefaultOptions(&allocator));
    ORT_CHECK(ort->SessionGetInputName(session, 0, allocator, &input_name));
    ORT_CHECK(ort->SessionGetOutputName(session, 0, allocator, &output_name));
    printf("Model input:  %s\n", input_name);
    printf("Model output: %s\n", output_name);

    /* --- allocate input tensor buffer --- */
    input_size    = 1 * 3 * MODEL_INPUT_H * MODEL_INPUT_W;
    input_data    = (float *)malloc(input_size * sizeof(float));
    if (!input_data) {
        fprintf(stderr, "Out of memory\n");
        goto cleanup;
    }

    input_dims[0] = 1;
    input_dims[1] = 3;
    input_dims[2] = MODEL_INPUT_H;
    input_dims[3] = MODEL_INPUT_W;

    ORT_CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    /* ===================================================================
     * Frame loop
     * =================================================================== */
    while (cap.read(frame)) {
        frame_idx++;
        if (frame_idx % 30 == 0)
            printf("Frame %d / %d\n", frame_idx, total);

        /* --- preprocess: resize and convert to float NCHW [0,1] --- */
        cv::resize(frame, resized,
                   cv::Size(MODEL_INPUT_W, MODEL_INPUT_H),
                   0, 0, cv::INTER_LINEAR);

        float scale_x = (float)frame_w / MODEL_INPUT_W;
        float scale_y = (float)frame_h / MODEL_INPUT_H;

        for (int row = 0; row < MODEL_INPUT_H; row++) {
            for (int col = 0; col < MODEL_INPUT_W; col++) {
                cv::Vec3b px        = resized.at<cv::Vec3b>(row, col);
                int       pixel_idx = row * MODEL_INPUT_W + col;
                /* OpenCV is BGR, YOLOv8 expects RGB — swap [0] and [2] */
                input_data[0 * MODEL_INPUT_H * MODEL_INPUT_W + pixel_idx] = px[2] / 255.0f; /* R */
                input_data[1 * MODEL_INPUT_H * MODEL_INPUT_W + pixel_idx] = px[1] / 255.0f; /* G */
                input_data[2 * MODEL_INPUT_H * MODEL_INPUT_W + pixel_idx] = px[0] / 255.0f; /* B */
            }
        }

        /* --- create input tensor and run inference --- */
        OrtValue *input_tensor  = NULL;
        OrtValue *output_tensor = NULL;

        ORT_CHECK(ort->CreateTensorWithDataAsOrtValue(
            mem_info, input_data, input_size * sizeof(float),
            input_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
        ));

        const char *input_names[]  = { input_name  };
        const char *output_names[] = { output_name };

        status = ort->Run(
            session, NULL,
            input_names,  (const OrtValue *const *)&input_tensor,  1,
            output_names, 1, &output_tensor
        );
        if (status != NULL) {
            fprintf(stderr, "Inference error: %s\n", ort->GetErrorMessage(status));
            ort->ReleaseStatus(status);
            status = NULL;
            ort->ReleaseValue(input_tensor);
            break;
        }

        /* --- parse output tensor [1, 84, 8400] --- */
        float *out_data = NULL;
        ORT_CHECK(ort->GetTensorMutableData(output_tensor, (void **)&out_data));

        Detection *dets    = (Detection *)malloc(NUM_PREDICTIONS * sizeof(Detection));
        int        num_dets = 0;

        for (int i = 0; i < NUM_PREDICTIONS; i++) {
            float cx = out_data[0 * NUM_PREDICTIONS + i];
            float cy = out_data[1 * NUM_PREDICTIONS + i];
            float bw = out_data[2 * NUM_PREDICTIONS + i];
            float bh = out_data[3 * NUM_PREDICTIONS + i];

            float best_score = 0.0f;
            int   best_class = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float s = out_data[(4 + c) * NUM_PREDICTIONS + i];
                if (s > best_score) { best_score = s; best_class = c; }
            }

            if (best_score < conf_thresh) continue;

            dets[num_dets].x        = cx;
            dets[num_dets].y        = cy;
            dets[num_dets].w        = bw;
            dets[num_dets].h        = bh;
            dets[num_dets].score    = best_score;
            dets[num_dets].class_id = best_class;
            dets[num_dets].keep     = 1;
            num_dets++;
        }

        /* --- sort by score descending --- */
        for (int i = 0; i < num_dets - 1; i++) {
            for (int j = i + 1; j < num_dets; j++) {
                if (dets[j].score > dets[i].score) {
                    Detection tmp = dets[i]; dets[i] = dets[j]; dets[j] = tmp;
                }
            }
        }

        /* --- NMS --- */
        for (int i = 0; i < num_dets; i++) {
            if (!dets[i].keep) continue;
            float box_i[4] = { dets[i].x, dets[i].y, dets[i].w, dets[i].h };
            for (int j = i + 1; j < num_dets; j++) {
                if (!dets[j].keep) continue;
                if (dets[j].class_id != dets[i].class_id) continue;
                float box_j[4] = { dets[j].x, dets[j].y, dets[j].w, dets[j].h };
                if (iou(box_i, box_j) > nms_thresh)
                    dets[j].keep = 0;
            }
        }

        /* --- draw boxes --- */
        for (int i = 0; i < num_dets; i++) {
            if (!dets[i].keep) continue;

            int x1 = (int)((dets[i].x - dets[i].w / 2) * scale_x);
            int y1 = (int)((dets[i].y - dets[i].h / 2) * scale_y);
            int x2 = (int)((dets[i].x + dets[i].w / 2) * scale_x);
            int y2 = (int)((dets[i].y + dets[i].h / 2) * scale_y);

            /* clamp to frame boundaries */
            x1 = x1 < 0 ? 0 : (x1 >= frame_w ? frame_w - 1 : x1);
            y1 = y1 < 0 ? 0 : (y1 >= frame_h ? frame_h - 1 : y1);
            x2 = x2 < 0 ? 0 : (x2 >= frame_w ? frame_w - 1 : x2);
            y2 = y2 < 0 ? 0 : (y2 >= frame_h ? frame_h - 1 : y2);

            cv::rectangle(frame,
                          cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);

            char label[64];
            snprintf(label, sizeof(label), "%s %.2f",
                     COCO_LABELS[dets[i].class_id], dets[i].score);

            int baseline;
            cv::Size text_size = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            cv::rectangle(frame,
                          cv::Point(x1, y1 - text_size.height - 4),
                          cv::Point(x1 + text_size.width, y1),
                          cv::Scalar(0, 255, 0), cv::FILLED);

            cv::putText(frame, label, cv::Point(x1, y1 - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1);
        }

        free(dets);
        writer.write(frame);
        ort->ReleaseValue(input_tensor);
        ort->ReleaseValue(output_tensor);
    }

cleanup:
    free(input_data);
    if (mem_info) ort->ReleaseMemoryInfo(mem_info);
    if (opts)     ort->ReleaseSessionOptions(opts);
    if (session)  ort->ReleaseSession(session);
    if (env)      ort->ReleaseEnv(env);
    cap.release();
    writer.release();

    printf("Done. Output saved to %s\n", output_path);
    return 0;
}