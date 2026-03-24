#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>
#include "coco_labels.h"

#define MODEL_INPUT_W   640
#define MODEL_INPUT_H   640
#define NUM_ANCHORS     8400   /* model output: [1, 84, 8400] */
#define NUM_OUTPUTS     84     /* 4 box coords + 80 COCO classes */
#define NMS_IOU_THRESH  0.45f

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
 * Letterbox transform descriptor
 * ----------------------------------------------------------------------- */
typedef struct {
    float pad_x;   /* left padding in model pixels */
    float pad_y;   /* top  padding in model pixels */
    float scale;   /* uniform scale applied to the original frame */
} LetterboxInfo;

/* -----------------------------------------------------------------------
 * Detection (post-NMS)
 * ----------------------------------------------------------------------- */
typedef struct {
    float x1, y1, x2, y2;  /* model-space coordinates */
    float score;
    int   cls;
} Detection;

/*
 * letterbox_into()
 *
 * Resizes `src` into a MODEL_INPUT_W x MODEL_INPUT_H canvas with grey
 * padding (114, 114, 114), preserving aspect ratio.
 * Fills `input_data` in NCHW RGB [0, 1] layout.
 * Returns a LetterboxInfo so detections can be mapped back to frame coords.
 */
static LetterboxInfo letterbox_into(const cv::Mat &src,
                                    float         *input_data,
                                    int            model_w,
                                    int            model_h)
{
    LetterboxInfo lb;

    float scale_w = (float)model_w / src.cols;
    float scale_h = (float)model_h / src.rows;
    lb.scale      = scale_w < scale_h ? scale_w : scale_h;

    int new_w = (int)roundf(src.cols * lb.scale);
    int new_h = (int)roundf(src.rows * lb.scale);

    lb.pad_x = (model_w - new_w) * 0.5f;
    lb.pad_y = (model_h - new_h) * 0.5f;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    /* fill canvas with grey (YOLOv8 standard: 114) */
    int   channel_size = model_w * model_h;
    float grey         = 114.0f / 255.0f;
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < channel_size; i++)
            input_data[c * channel_size + i] = grey;

    /* copy resized pixels into padded canvas — NCHW, RGB */
    int pad_x = (int)roundf(lb.pad_x);
    int pad_y = (int)roundf(lb.pad_y);

    for (int row = 0; row < new_h; row++) {
        const cv::Vec3b *src_row  = resized.ptr<cv::Vec3b>(row);
        int              dst_row  = pad_y + row;
        for (int col = 0; col < new_w; col++) {
            int dst_col   = pad_x + col;
            int pixel_idx = dst_row * model_w + dst_col;
            input_data[0 * channel_size + pixel_idx] = src_row[col][2] / 255.0f; /* R */
            input_data[1 * channel_size + pixel_idx] = src_row[col][1] / 255.0f; /* G */
            input_data[2 * channel_size + pixel_idx] = src_row[col][0] / 255.0f; /* B */
        }
    }

    return lb;
}

/* -----------------------------------------------------------------------
 * IoU between two detections (model-space coords)
 * ----------------------------------------------------------------------- */
static float iou_calc(const Detection *a, const Detection *b)
{
    float ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float iy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float iw  = ix2 - ix1;
    float ih  = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    float inter  = iw * ih;
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    return inter / (area_a + area_b - inter);
}

/* -----------------------------------------------------------------------
 * Greedy cross-class NMS.  Sorts `dets` in-place by score desc.
 * Returns the number of kept detections (packed at the front of `dets`).
 * ----------------------------------------------------------------------- */
static int nms(Detection *dets, int n, float iou_thresh)
{
    /* insertion sort by score descending */
    for (int i = 1; i < n; i++) {
        Detection tmp = dets[i];
        int j = i - 1;
        while (j >= 0 && dets[j].score < tmp.score) {
            dets[j + 1] = dets[j];
            j--;
        }
        dets[j + 1] = tmp;
    }

    int *suppressed = (int *)calloc(n, sizeof(int));
    if (!suppressed) return 0;

    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;
        dets[kept++] = dets[i];
        for (int j = i + 1; j < n; j++) {
            if (!suppressed[j] && iou_calc(&dets[i], &dets[j]) > iou_thresh)
                suppressed[j] = 1;
        }
    }

    free(suppressed);
    return kept;
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    OrtStatus         *status      = NULL;
    OrtEnv            *env         = NULL;
    OrtSessionOptions *opts        = NULL;
    OrtSession        *session     = NULL;
    OrtAllocator      *allocator   = NULL;
    OrtMemoryInfo     *mem_info    = NULL;
    char              *input_name  = NULL;
    char              *output_name = NULL;
    float             *input_data  = NULL;
    Detection         *raw_dets    = NULL;
    int64_t            input_dims[4];
    size_t             input_size  = 0;
    int                ret         = 0;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.mp4> <model.onnx> <output.mp4> [conf_thresh]\n",
                argv[0]);
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

    OrtCUDAProviderOptions cuda_opts;
    memset(&cuda_opts, 0, sizeof(cuda_opts));
    cuda_opts.device_id = 0;

    ORT_CHECK(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "detect", &env));
    ORT_CHECK(ort->CreateSessionOptions(&opts));
    ORT_CHECK(ort->SessionOptionsAppendExecutionProvider_CUDA(opts, &cuda_opts));
    ORT_CHECK(ort->CreateSession(env, model_path, opts, &session));
    printf("Model loaded.\n");

    ORT_CHECK(ort->GetAllocatorWithDefaultOptions(&allocator));
    ORT_CHECK(ort->SessionGetInputName(session, 0, allocator, &input_name));
    ORT_CHECK(ort->SessionGetOutputName(session, 0, allocator, &output_name));
    printf("Model input:  %s\n", input_name);
    printf("Model output: %s\n", output_name);

    /* --- allocate buffers --- */
    input_size = 1 * 3 * MODEL_INPUT_H * MODEL_INPUT_W;
    input_data = (float *)malloc(input_size * sizeof(float));
    if (!input_data) { fprintf(stderr, "Out of memory\n"); goto cleanup; }

    raw_dets = (Detection *)malloc(NUM_ANCHORS * sizeof(Detection));
    if (!raw_dets) { fprintf(stderr, "Out of memory\n"); goto cleanup; }

    input_dims[0] = 1;
    input_dims[1] = 3;
    input_dims[2] = MODEL_INPUT_H;
    input_dims[3] = MODEL_INPUT_W;

    ORT_CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    /* --- C++ objects in a nested scope so destructors fire before cleanup --- */
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

        cv::Mat       frame;
        int           frame_idx = 0;
        LetterboxInfo lb;

        while (cap.read(frame)) {
            frame_idx++;
            if (frame_idx % 100 == 0)
                printf("Frame %d / %d\n", frame_idx, total);

            /* -----------------------------------------------------------------
             * Preprocess: letterbox to 640x640, NCHW RGB [0,1]
             * ----------------------------------------------------------------- */
            lb = letterbox_into(frame, input_data, MODEL_INPUT_W, MODEL_INPUT_H);

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
             * Parse output [1, 84, 8400]:
             *   channels 0-3  : cx, cy, w, h  (model-space pixels)
             *   channels 4-83 : class scores  (already sigmoid'd by model)
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

            int n_raw = 0;
            for (int i = 0; i < NUM_ANCHORS; i++) {
                /* find max class score across 80 classes */
                float best_score = -1.0f;
                int   best_cls   =  0;
                for (int c = 4; c < NUM_OUTPUTS; c++) {
                    float s = out_data[c * NUM_ANCHORS + i];
                    if (s > best_score) { best_score = s; best_cls = c - 4; }
                }
                if (best_score < conf_thresh) continue;

                float cx = out_data[0 * NUM_ANCHORS + i];
                float cy = out_data[1 * NUM_ANCHORS + i];
                float w  = out_data[2 * NUM_ANCHORS + i];
                float h  = out_data[3 * NUM_ANCHORS + i];

                raw_dets[n_raw].x1    = cx - w * 0.5f;
                raw_dets[n_raw].y1    = cy - h * 0.5f;
                raw_dets[n_raw].x2    = cx + w * 0.5f;
                raw_dets[n_raw].y2    = cy + h * 0.5f;
                raw_dets[n_raw].score = best_score;
                raw_dets[n_raw].cls   = best_cls;
                n_raw++;
            }

            int n_kept = nms(raw_dets, n_raw, NMS_IOU_THRESH);

            for (int i = 0; i < n_kept; i++) {
                if (raw_dets[i].cls < 0 || raw_dets[i].cls >= NUM_CLASSES) continue;

                /* undo letterbox: subtract padding, divide by uniform scale */
                int bx1 = (int)((raw_dets[i].x1 - lb.pad_x) / lb.scale);
                int by1 = (int)((raw_dets[i].y1 - lb.pad_y) / lb.scale);
                int bx2 = (int)((raw_dets[i].x2 - lb.pad_x) / lb.scale);
                int by2 = (int)((raw_dets[i].y2 - lb.pad_y) / lb.scale);

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
                snprintf(label, sizeof(label), "%s %.2f",
                         COCO_LABELS[raw_dets[i].cls], raw_dets[i].score);

                /* draw label background */
                int      baseline  = 0;
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
    free(raw_dets);
    free(input_data);
    if (mem_info) ort->ReleaseMemoryInfo(mem_info);
    if (opts)     ort->ReleaseSessionOptions(opts);
    if (session)  ort->ReleaseSession(session);
    if (env)      ort->ReleaseEnv(env);

    printf("Done. Output saved to %s\n", output_path);
    return ret;
}
