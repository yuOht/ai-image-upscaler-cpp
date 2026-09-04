// ai_scale_refactored.cpp
// OpenVINOを用いたタイル分割AI超解像処理
//
// リファクタリング内容:
//  - HWC<->CHWのテンソル変換処理を共通関数化 (matToChwTensor / chwTensorToMat)
//  - cv::Mat::at<>() によるピクセルアクセスを生ポインタ直接アクセスに変更（高速化）
//  - 推論デバイスを "GPU" 固定から自動フォールバック (GPU失敗時はCPUで再試行) に変更
//  - マジックナンバーを名前付き定数として整理
//  - 入力引数のバリデーションを追加

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <exception>
#include <vector>
#include <map>
#include <memory>

namespace
{
    // タイル処理のパラメータ
    constexpr int kTileSize = 64;
    constexpr int kTileMargin = 16;

    // AI出力を 0-255 とみなすか 0-1 とみなすかの閾値
    // (モデルによって出力レンジが異なるため、輝度の最大値で自動判定する)
    constexpr double kNormalizedOutputThreshold = 2.0;

    // 入力画像を [0,1] のfloatに正規化してから CHW 形式の連続バッファへ変換する。
    // 戻り値のバッファは ov::Tensor が参照するため、Tensor を使い終わるまで
    // 呼び出し側で生存させておくこと。
    std::vector<float> matToChwTensor(const cv::Mat &bgrFloat)
    {
        const int h = bgrFloat.rows;
        const int w = bgrFloat.cols;
        std::vector<float> chw(static_cast<size_t>(3) * h * w);

        const size_t plane = static_cast<size_t>(h) * w;
        for (int y = 0; y < h; y++)
        {
            const float *row = bgrFloat.ptr<float>(y);
            float *rB = chw.data() + static_cast<size_t>(y) * w;
            float *rG = chw.data() + plane + static_cast<size_t>(y) * w;
            float *rR = chw.data() + 2 * plane + static_cast<size_t>(y) * w;
            for (int x = 0; x < w; x++)
            {
                rB[x] = row[x * 3 + 0];
                rG[x] = row[x * 3 + 1];
                rR[x] = row[x * 3 + 2];
            }
        }
        return chw;
    }

    // OpenVINOの出力テンソル(CHW, float)を CV_32FC3 の cv::Mat (BGR) に変換する。
    cv::Mat chwTensorToMat(const float *data, int outH, int outW)
    {
        cv::Mat out(outH, outW, CV_32FC3);
        const size_t plane = static_cast<size_t>(outH) * outW;
        const float *pB = data;
        const float *pG = data + plane;
        const float *pR = data + 2 * plane;

        for (int y = 0; y < outH; y++)
        {
            float *row = out.ptr<float>(y);
            const float *rB = pB + static_cast<size_t>(y) * outW;
            const float *rG = pG + static_cast<size_t>(y) * outW;
            const float *rR = pR + static_cast<size_t>(y) * outW;
            for (int x = 0; x < outW; x++)
            {
                row[x * 3 + 0] = rB[x];
                row[x * 3 + 1] = rG[x];
                row[x * 3 + 2] = rR[x];
            }
        }
        return out;
    }

    // AIモデルの出力を 0-255 の CV_8UC3 に安全にスケーリングする。
    // モデルによって出力レンジが [0,1] のものと [0,255] のものが混在するため、
    // 輝度の最大値を見て自動判定する。
    cv::Mat scaleOutputTo8U(const cv::Mat &outputFloat)
    {
        cv::Mat gray, output8u;
        cv::cvtColor(outputFloat, gray, cv::COLOR_BGR2GRAY);
        double minVal, maxVal;
        cv::minMaxLoc(gray, &minVal, &maxVal);

        if (maxVal <= kNormalizedOutputThreshold)
        {
            // 出力が 0〜1.0 の範囲だった場合は 255 倍して戻す
            outputFloat.convertTo(output8u, CV_8UC3, 255.0);
        }
        else
        {
            // 出力がすでに 0〜255 の範囲だった場合はそのままキャスト
            outputFloat.convertTo(output8u, CV_8UC3);
        }
        return output8u;
    }

    // GPUでのコンパイルを試み、失敗した場合はCPUにフォールバックする。
    ov::CompiledModel compileWithFallback(ov::Core &core, const std::shared_ptr<ov::Model> &model)
    {
        try
        {
            return core.compile_model(model, "GPU");
        }
        catch (const std::exception &e)
        {
            std::cerr << "[警告] GPUでのモデルコンパイルに失敗したためCPUにフォールバックします: "
                      << e.what() << std::endl;
            return core.compile_model(model, "CPU");
        }
    }
}

cv::Mat infer_tile_openvino(
    ov::InferRequest &infer_request,
    const cv::Mat &input_tile,
    int scale)
{
    try
    {
        cv::Mat inputFloat;
        // Real-ESRGAN系モデル向けに 1/255 正規化
        input_tile.convertTo(inputFloat, CV_32F, 1.0 / 255.0);
        // 入力はOpenCVと同じBGR順のままモデルに渡す（Intel OMZモデル準拠）

        const int h = inputFloat.rows;
        const int w = inputFloat.cols;

        std::vector<float> inputData = matToChwTensor(inputFloat);
        ov::Tensor inputTensor(ov::element::f32, {1, 3, (size_t)h, (size_t)w}, inputData.data());
        infer_request.set_input_tensor(0, inputTensor);

        // 2入力モデル（バイキュービック補間画像も入力に使うタイプ）への対応
        std::vector<float> inputData2;
        if (infer_request.get_compiled_model().inputs().size() >= 2)
        {
            cv::Mat bicubicFloat;
            cv::resize(inputFloat, bicubicFloat, cv::Size(w * scale, h * scale), 0, 0, cv::INTER_CUBIC);

            inputData2 = matToChwTensor(bicubicFloat);
            ov::Tensor inputTensor2(ov::element::f32,
                                    {1, 3, (size_t)(h * scale), (size_t)(w * scale)},
                                    inputData2.data());
            infer_request.set_input_tensor(1, inputTensor2);
        }

        infer_request.infer();

        auto outputTensor = infer_request.get_output_tensor();
        auto shape = outputTensor.get_shape();
        const int outH = static_cast<int>(shape[2]);
        const int outW = static_cast<int>(shape[3]);

        cv::Mat output = chwTensorToMat(outputTensor.data<const float>(), outH, outW);

        return scaleOutputTo8U(output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n[OpenVINO 推論エラー] " << e.what() << std::endl;
        return cv::Mat();
    }
}

extern "C"
{
    unsigned char *ScaleImageAI_OpenCV(unsigned char *inputData, int width, int height, int scale,
                                       int *outWidth, int *outHeight, const char *modelPath)
    {
        if (inputData == nullptr || width <= 0 || height <= 0 || scale <= 0 ||
            outWidth == nullptr || outHeight == nullptr || modelPath == nullptr)
        {
            std::cerr << "[エラー] ScaleImageAI_OpenCV: 不正な引数です。" << std::endl;
            return nullptr;
        }

        cv::Mat img(height, width, CV_8UC3, inputData);
        cv::Mat bgr_img;
        cv::cvtColor(img, bgr_img, cv::COLOR_RGB2BGR);

        ov::Core core;
        ov::CompiledModel compiledModel;
        ov::InferRequest inferRequest;

        try
        {
            std::shared_ptr<ov::Model> model = core.read_model(modelPath);

            if (model->inputs().size() == 1)
            {
                model->reshape({1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
            }
            else if (model->inputs().size() >= 2)
            {
                std::map<size_t, ov::PartialShape> shapes;
                shapes[0] = {1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()};
                shapes[1] = {1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()};
                model->reshape(shapes);
            }

            compiledModel = compileWithFallback(core, model);
            inferRequest = compiledModel.create_infer_request();
        }
        catch (const std::exception &e)
        {
            std::cerr << "\n[エラー] AIモデルの読み込み・コンパイルに失敗しました: " << e.what() << std::endl;
            return nullptr;
        }

        cv::setNumThreads(10);

        *outWidth = width * scale;
        *outHeight = height * scale;

        cv::Mat result(*outHeight, *outWidth, CV_8UC3);

        std::cout << "[AI超解像] モデル推論を実行中..." << std::endl;

        const int totalTiles =
            ((width + kTileSize - 1) / kTileSize) * ((height + kTileSize - 1) / kTileSize);
        int currentTile = 0;

        for (int y = 0; y < height; y += kTileSize)
        {
            for (int x = 0; x < width; x += kTileSize)
            {
                const int currentWidth = std::min(kTileSize, width - x);
                const int currentHeight = std::min(kTileSize, height - y);

                const int x_start = std::max(0, x - kTileMargin);
                const int y_start = std::max(0, y - kTileMargin);
                const int x_end = std::min(width, x + currentWidth + kTileMargin);
                const int y_end = std::min(height, y + currentHeight + kTileMargin);

                const int exp_w = x_end - x_start;
                const int exp_h = y_end - y_start;

                cv::Rect srcRoiWithMargin(x_start, y_start, exp_w, exp_h);
                cv::Mat tile = bgr_img(srcRoiWithMargin);

                cv::Mat upsampledTile = infer_tile_openvino(inferRequest, tile, scale);

                if (upsampledTile.empty())
                {
                    std::cerr << "\n[エラー] タイルのAI処理に失敗したため中断します。" << std::endl;
                    return nullptr;
                }

                const int offset_x = (x - x_start) * scale;
                const int offset_y = (y - y_start) * scale;
                cv::Rect validRoi(offset_x, offset_y, currentWidth * scale, currentHeight * scale);

                if (validRoi.x + validRoi.width > upsampledTile.cols ||
                    validRoi.y + validRoi.height > upsampledTile.rows)
                {
                    std::cerr << "\n[領域エラー] クロッピング範囲がAI出力サイズを超過しています。" << std::endl;
                    return nullptr;
                }

                cv::Mat croppedTile = upsampledTile(validRoi);
                cv::Rect dstRoi(x * scale, y * scale, currentWidth * scale, currentHeight * scale);
                croppedTile.copyTo(result(dstRoi));

                currentTile++;
                if (currentTile % 10 == 0 || currentTile == totalTiles)
                {
                    std::cout << "  -> 進捗: " << currentTile << " / " << totalTiles << " タイル完了\r" << std::flush;
                }
            }
        }
        std::cout << std::endl
                  << "[AI超解像] 推論完了！" << std::endl;

        cv::Mat rgb_result;
        cv::cvtColor(result, rgb_result, cv::COLOR_BGR2RGB);

        const size_t dataSize = static_cast<size_t>(*outWidth) * (*outHeight) * 3;
        unsigned char *outputData = static_cast<unsigned char *>(malloc(dataSize));
        if (outputData != nullptr)
        {
            std::memcpy(outputData, rgb_result.data, dataSize);
        }
        else
        {
            std::cerr << "[エラー] 出力バッファの確保に失敗しました。" << std::endl;
        }

        return outputData;
    }
} // extern "C"