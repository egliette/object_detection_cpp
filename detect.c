#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>
#include "coco_labels.h"

#define MODEL_INPUT_W   640
#define MODEL_INPUT_H   640
#define MAX_DETECTIONS  300  /* model output: [1, 300, 6] */

/* ORT_CHECK: only use outside the frame loop.
 * Inside the loop we check manually so we can break cleanly. */
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
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* --- all POD variables declared at top so goto cleanup is always safe --- */
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
    size_t             input_size  = 0;
    int                ret         = 0;

    /* --- argument parsing --- */
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.mp4> <model.onnx> <output.mp4> [conf_thresh]\n", argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *model_path  = argv[2];
    const char *output_path = argv[3];
    float conf_thresh = (argc >= 5) ? (float)atof(argv[4]) : 0.45f;

    printf("Input:  %s\n", input_path);
    printf("Model:  %s\n", model_path);
    printf("Output: %s\n", output_path);
    printf("Conf threshold: %.2f\n", conf_thresh);

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

    /* --- allocate input buffer: [1, 3, 640, 640] float32 --- */
    input_size   = 1 * 3 * MODEL_INPUT_H * MODEL_INPUT_W;
    input_data   = (float *)malloc(input_size * sizeof(float));
    if (!input_data) { fprintf(stderr, "Out of memory\n"); goto cleanup; }

    input_dims[0] = 1;
    input_dims[1] = 3;
    input_dims[2] = MODEL_INPUT_H;
    input_dims[3] = MODEL_INPUT_W;

    ORT_CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    /* --- C++ objects in a nested scope so their destructors fire before cleanup --- */
    {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            fprintf(stderr, "Error: cannot open input video: %s\n", input_path);
            ret = 1; goto cleanup;
        }

        int    frame_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int    frame_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps     = cap.get(cv::CAP_PROP_FPS);
        int    total   = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        printf("Video: %dx%d @ %.1f fps, %d frames\n", frame_w, frame_h, fps, total);

        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(output_path, fourcc, fps, cv::Size(frame_w, frame_h));
        if (!writer.isOpened()) {
            fprintf(stderr, "Error: cannot open output video: %s\n", output_path);
            ret = 1; goto cleanup;
        }

        cv::Mat frame;
        cv::Mat resized;
        int frame_idx = 0;

        /* scale factors: map 640x640 model coords → original frame pixels */
        float scale_x = (float)frame_w / MODEL_INPUT_W;
        float scale_y = (float)frame_h / MODEL_INPUT_H;

        while (cap.read(frame)) {
            frame_idx++;
            if (frame_idx % 30 == 0)
                printf("Frame %d / %d\n", frame_idx, total);

            /* -----------------------------------------------------------------
             * Preprocess: resize to 640x640, convert BGR→RGB, normalize to [0,1]
             * Layout: NCHW — all R pixels, then all G, then all B
             * ----------------------------------------------------------------- */
            cv::resize(frame, resized,
                       cv::Size(MODEL_INPUT_W, MODEL_INPUT_H),
                       0, 0, cv::INTER_LINEAR);

            for (int row = 0; row < MODEL_INPUT_H; row++) {
                for (int col = 0; col < MODEL_INPUT_W; col++) {
                    cv::Vec3b px        = resized.at<cv::Vec3b>(row, col);
                    int       pixel_idx = row * MODEL_INPUT_W + col;
                    input_data[0 * MODEL_INPUT_H * MODEL_INPUT_W + pixel_idx] = px[2] / 255.0f; /* R */
                    input_data[1 * MODEL_INPUT_H * MODEL_INPUT_W + pixel_idx] = px[1] / 255.0f; /* G */
                    input_data[2 * MODEL_INPUT_H * MODEL_INPUT_W + pixel_idx] = px[0] / 255.0f; /* B */
                }
            }

            /* -----------------------------------------------------------------
             * Create input tensor (zero-copy — wraps input_data directly)
             * ----------------------------------------------------------------- */
            OrtValue *input_tensor  = NULL;
            OrtValue *output_tensor = NULL;

            status = ort->CreateTensorWithDataAsOrtValue(
                mem_info, input_data, input_size * sizeof(float),
                input_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &input_tensor
            );
            if (status != NULL) {
                fprintf(stderr, "CreateTensor error: %s\n", ort->GetErrorMessage(status));
                ort->ReleaseStatus(status); status = NULL;
                break;
            }

            /* -----------------------------------------------------------------
             * Run inference
             * ----------------------------------------------------------------- */
            const char *input_names[]  = { input_name  };
            const char *output_names[] = { output_name };

            status = ort->Run(
                session, NULL,
                input_names,  (const OrtValue *const *)&input_tensor, 1,
                output_names, 1, &output_tensor
            );
            if (status != NULL) {
                fprintf(stderr, "Inference error: %s\n", ort->GetErrorMessage(status));
                ort->ReleaseStatus(status); status = NULL;
                ort->ReleaseValue(input_tensor);
                break;
            }

            /* -----------------------------------------------------------------
             * Parse output tensor: [1, 300, 6]
             * Each detection: [x1, y1, x2, y2, score, class_id]
             * Coordinates are in 640x640 model space.
             * NMS is already done by the model — no manual NMS needed.
             * ----------------------------------------------------------------- */
            float *out_data = NULL;
            status = ort->GetTensorMutableData(output_tensor, (void **)&out_data);
            if (status != NULL) {
                fprintf(stderr, "GetTensorData error: %s\n", ort->GetErrorMessage(status));
                ort->ReleaseStatus(status); status = NULL;
                ort->ReleaseValue(input_tensor);
                ort->ReleaseValue(output_tensor);
                break;
            }

            for (int i = 0; i < MAX_DETECTIONS; i++) {
                float x1    = out_data[i * 6 + 0];
                float y1    = out_data[i * 6 + 1];
                float x2    = out_data[i * 6 + 2];
                float y2    = out_data[i * 6 + 3];
                float score = out_data[i * 6 + 4];
                int   cls   = (int)out_data[i * 6 + 5];

                /* model pads unused slots with zeros — skip them */
                if (score < conf_thresh) continue;
                if (cls < 0 || cls >= NUM_CLASSES) continue;

                /* scale from 640x640 → original frame size */
                int bx1 = (int)(x1 * scale_x);
                int by1 = (int)(y1 * scale_y);
                int bx2 = (int)(x2 * scale_x);
                int by2 = (int)(y2 * scale_y);

                /* clamp to frame boundaries */
                bx1 = bx1 < 0 ? 0 : (bx1 >= frame_w ? frame_w - 1 : bx1);
                by1 = by1 < 0 ? 0 : (by1 >= frame_h ? frame_h - 1 : by1);
                bx2 = bx2 < 0 ? 0 : (bx2 >= frame_w ? frame_w - 1 : bx2);
                by2 = by2 < 0 ? 0 : (by2 >= frame_h ? frame_h - 1 : by2);

                /* draw bounding box */
                cv::rectangle(frame,
                              cv::Point(bx1, by1), cv::Point(bx2, by2),
                              cv::Scalar(0, 255, 0), 2);

                /* build label: "person 0.87" */
                char label[64];
                snprintf(label, sizeof(label), "%s %.2f", COCO_LABELS[cls], score);

                /* draw label background */
                int baseline;
                cv::Size text_size = cv::getTextSize(
                    label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(frame,
                              cv::Point(bx1, by1 - text_size.height - 4),
                              cv::Point(bx1 + text_size.width, by1),
                              cv::Scalar(0, 255, 0), cv::FILLED);

                /* draw label text */
                cv::putText(frame, label, cv::Point(bx1, by1 - 2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 0, 0), 1);
            }

            writer.write(frame);
            ort->ReleaseValue(input_tensor);
            ort->ReleaseValue(output_tensor);
        }
        /* cap and writer destructors run here */
    }

cleanup:
    free(input_data);
    if (mem_info) ort->ReleaseMemoryInfo(mem_info);
    if (opts)     ort->ReleaseSessionOptions(opts);
    if (session)  ort->ReleaseSession(session);
    if (env)      ort->ReleaseEnv(env);

    printf("Done. Output saved to %s\n", output_path);
    return ret;
}