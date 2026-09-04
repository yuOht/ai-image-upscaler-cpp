// scaling_refactored.c
// 画像を拡大後、アンシャープマスキングを適用してシャープ化するプログラム
//
// リファクタリング内容:
//  - AI超解像のモデルパスをハードコードから「コマンドライン引数 or 環境変数」に変更
//  - 3種類の補間サンプリング関数に共通する境界クランプ処理を ClampLong() として共通化
//  - 補間モードのディスパッチを if-else の羅列から関数ポインタテーブルに変更
//  - malloc の失敗チェックを追加
//  - 使い方メッセージにAIモデルパス引数を追記

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// AI超解像関数の外部宣言
#ifdef __cplusplus
extern "C"
{
#endif

    extern unsigned char *ScaleImageAI_OpenCV(unsigned char *inputData, int width, int height, int scale, int *outWidth, int *outHeight, const char *modelPath);

#ifdef __cplusplus
}
#endif

// stb_image ライブラリの読み込み (JPEG対応)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define ROUND(x) ((long)((x) + 0.5))

// AIモデルパスを指定しなかった場合に参照する環境変数名
#define AI_MODEL_ENV_VAR "ESRGAN_MODEL_PATH"

// --------------------------------------------------------------------
// 共通ユーティリティ
// --------------------------------------------------------------------

// 値を [0, maxExclusive-1] の範囲にクランプする
static long ClampLong(long v, long maxExclusive)
{
    if (v < 0)
        return 0;
    if (v >= maxExclusive)
        return maxExclusive - 1;
    return v;
}

// --------------------------------------------------------------------
// 補間アルゴリズム
// --------------------------------------------------------------------

// 1. ニアレストネイバー法
unsigned char NearestNeighborSample(unsigned char *image, long width, long height, double x, double y, int c)
{
    long ix = ClampLong(ROUND(x), width);
    long iy = ClampLong(ROUND(y), height);
    return image[(iy * width + ix) * 3 + c];
}

// 2. バイリニア法
unsigned char BilinearSample(unsigned char *image, long width, long height, double x, double y, int c)
{
    long ix = (long)floor(x);
    long iy = (long)floor(y);
    double dx = x - ix;
    double dy = y - iy;

    long x1 = ClampLong(ix, width);
    long y1 = ClampLong(iy, height);
    long x2 = ClampLong(ix + 1, width);
    long y2 = ClampLong(iy + 1, height);

    double val = (1.0 - dx) * (1.0 - dy) * image[(y1 * width + x1) * 3 + c] +
                 dx * (1.0 - dy) * image[(y1 * width + x2) * 3 + c] +
                 (1.0 - dx) * dy * image[(y2 * width + x1) * 3 + c] +
                 dx * dy * image[(y2 * width + x2) * 3 + c];

    return (unsigned char)ROUND(val);
}

// 3. バイキュービック法の重み計算
double CalculateWeight(double t)
{
    const double a = -1.0;
    double absT = fabs(t);

    if (absT <= 1.0)
    {
        return (a + 2.0) * absT * absT * absT - (a + 3.0) * absT * absT + 1.0;
    }
    else if (absT <= 2.0)
    {
        return a * absT * absT * absT - 5.0 * a * absT * absT + 8.0 * a * absT - 4.0 * a;
    }
    return 0.0;
}

// 3. バイキュービック法
unsigned char BicubicSample(unsigned char *image, long width, long height, double x, double y, int c)
{
    long ix, iy, jx, jy, px, py;
    double dx, dy, wx, wy, weight, sum, weightSum, value;

    if (x < 0.0 || y < 0.0 || x > (double)(width - 1) || y > (double)(height - 1))
        return 0;

    ix = (long)floor(x);
    iy = (long)floor(y);
    dx = x - (double)ix;
    dy = y - (double)iy;

    sum = 0.0;
    weightSum = 0.0;
    for (jy = -1; jy <= 2; jy++)
    {
        for (jx = -1; jx <= 2; jx++)
        {
            px = ClampLong(ix + jx, width);
            py = ClampLong(iy + jy, height);

            wx = CalculateWeight((double)jx - dx);
            wy = CalculateWeight((double)jy - dy);
            weight = wx * wy;

            sum += weight * (double)image[(py * width + px) * 3 + c];
            weightSum += weight;
        }
    }

    value = sum / weightSum;
    if (value < 0.0)
        value = 0.0;
    if (value > 255.0)
        value = 255.0;

    return (unsigned char)ROUND(value);
}

// 補間関数のシグネチャと、モード番号からのディスパッチテーブル
typedef unsigned char (*SampleFunc)(unsigned char *, long, long, double, double, int);

static const SampleFunc kSampleFuncs[3] = {
    NearestNeighborSample,
    BilinearSample,
    BicubicSample,
};

static const char *const kMethodNames[3] = {
    "ニアレストネイバー法",
    "バイリニア法",
    "バイキュービック法",
};

// --------------------------------------------------------------------
// 1. 画像拡大処理 (従来の補間方式)
// --------------------------------------------------------------------
unsigned char *ScaleImage(unsigned char *inputImage, long width, long height, double scale, int method, long *outWidth, long *outHeight)
{
    if (method < 0 || method > 2)
    {
        fprintf(stderr, "エラー: 補間モードは 0, 1, 2 のいずれかを指定してください。\n");
        exit(EXIT_FAILURE);
    }

    *outWidth = (long)ROUND(width * scale);
    *outHeight = (long)ROUND(height * scale);

    unsigned char *scaledImage = (unsigned char *)malloc((size_t)(*outWidth) * (*outHeight) * 3);
    if (scaledImage == NULL)
    {
        fprintf(stderr, "メモリの確保に失敗しました\n");
        exit(EXIT_FAILURE);
    }

    printf("[拡大] モード指定: %s (並列処理中...)\n", kMethodNames[method]);
    SampleFunc sample = kSampleFuncs[method];

// OpenMPによる並列化 (行単位でスレッドに分割)
#pragma omp parallel for
    for (long i = 0; i < *outHeight; i++)
    {
        for (long j = 0; j < *outWidth; j++)
        {
            double srcX = (double)j / scale;
            double srcY = (double)i / scale;

            for (int c = 0; c < 3; c++)
            {
                long index = (i * (*outWidth) + j) * 3 + c;
                scaledImage[index] = sample(inputImage, width, height, srcX, srcY, c);
            }
        }
    }
    return scaledImage;
}

// --------------------------------------------------------------------
// 2. シャープ化処理（整数演算による超高速化版）
// --------------------------------------------------------------------
unsigned char *SharpenImage(unsigned char *inputImage, long width, long height, int filterSize, double k)
{
    if (k <= 0.0)
    {
        printf("[シャープ化] 強度k=0のためシャープ化をスキップします\n");
        return inputImage;
    }

    printf("[シャープ化] アンシャープマスキング適用 (FilterSize=%d, k=%.2f) (整数演算・並列処理中...)\n", filterSize, k);

    unsigned char *sharpenedImage = (unsigned char *)malloc((size_t)width * height * 3);
    if (sharpenedImage == NULL)
    {
        fprintf(stderr, "メモリの確保に失敗しました\n");
        exit(EXIT_FAILURE);
    }

    int halfFilter = filterSize / 2;
    double numCells = (double)(filterSize * filterSize);

    // --- 固定小数点演算のための事前準備 ---
    // 浮動小数点の重みを 1024倍 (2^10) して整数化する
    const int FIXED_POINT_SHIFT = 10;
    const int MULTIPLIER = 1 << FIXED_POINT_SHIFT; // 1024

    int centerWeightInt = (int)((1.0 + (numCells - 1.0) * k / numCells) * MULTIPLIER);
    int edgeWeightInt = (int)((-k / numCells) * MULTIPLIER);
    // ----------------------------------------
    // OpenMPによる並列化 (行単位でスレッドに分割)
#pragma omp parallel for
    for (long i = 0; i < height; i++)
    {
        for (long j = 0; j < width; j++)
        {
            if (i < halfFilter || i >= height - halfFilter || j < halfFilter || j >= width - halfFilter)
            {
                // 画像端部はメモリコピーを一括で行う (ループより高速)
                memcpy(&sharpenedImage[(i * width + j) * 3], &inputImage[(i * width + j) * 3], 3);
                continue;
            }

            // RGBそれぞれの合計値を保持する整数変数
            int sumR = 0, sumG = 0, sumB = 0;

            for (int di = -halfFilter; di <= halfFilter; di++)
            {
                for (int dj = -halfFilter; dj <= halfFilter; dj++)
                {
                    // フィルタの中心か否かで重みを切り替え
                    int weight = (di == 0 && dj == 0) ? centerWeightInt : edgeWeightInt;
                    long idx = ((i + di) * width + (j + dj)) * 3;

                    // double型を使わず、完全に整数のみで掛け算と足し算を行う
                    sumR += inputImage[idx] * weight;
                    sumG += inputImage[idx + 1] * weight;
                    sumB += inputImage[idx + 2] * weight;
                }
            }

            // 1024倍されているので、10ビット右シフトして元に戻す (割り算より圧倒的に高速)
            // 四捨五入のためのオフセット (1024 / 2 = 512) を足してからシフトする
            int outR = (sumR + (MULTIPLIER >> 1)) >> FIXED_POINT_SHIFT;
            int outG = (sumG + (MULTIPLIER >> 1)) >> FIXED_POINT_SHIFT;
            int outB = (sumB + (MULTIPLIER >> 1)) >> FIXED_POINT_SHIFT;

            // クランプ処理 (0~255に収める)
            sharpenedImage[(i * width + j) * 3] = (unsigned char)(outR < 0 ? 0 : (outR > 255 ? 255 : outR));
            sharpenedImage[(i * width + j) * 3 + 1] = (unsigned char)(outG < 0 ? 0 : (outG > 255 ? 255 : outG));
            sharpenedImage[(i * width + j) * 3 + 2] = (unsigned char)(outB < 0 ? 0 : (outB > 255 ? 255 : outB));
        }
    }
    return sharpenedImage;
}

char *MakeOutputFileName(const char *inputFile, const char *suffix)
{
    char *baseName = (char *)malloc(strlen(inputFile) + 1);
    if (baseName == NULL)
        exit(EXIT_FAILURE);
    strcpy(baseName, inputFile);

    char *pDot = strrchr(baseName, '.');
    if (pDot != NULL)
        *pDot = '\0';

    size_t baseLen = strlen(baseName);
    size_t suffixLen = strlen(suffix);
    char *outputFile = (char *)malloc(baseLen + suffixLen + 1);
    if (outputFile == NULL)
        exit(EXIT_FAILURE);

    strcpy(outputFile, baseName);
    strcat(outputFile, suffix);

    free(baseName);
    return outputFile;
}

// AI超解像用のモデルパスを決定する。
// 優先順位: コマンドライン引数(argv[5]) > 環境変数 ESRGAN_MODEL_PATH。
// どちらも無ければ NULL を返す。
static const char *ResolveModelPath(int argc, char *argv[])
{
    if (argc >= 6 && argv[5][0] != '\0')
        return argv[5];

    const char *envPath = getenv(AI_MODEL_ENV_VAR);
    if (envPath != NULL && envPath[0] != '\0')
        return envPath;

    return NULL;
}

static void PrintUsage(const char *progName)
{
    fprintf(stderr, "使い方: %s 入力ファイル.jpg(または.png等) 拡大率 補間モード シャープ化強度k [AIモデルパス.xml]\n", progName);
    fprintf(stderr, "  [補間モード]\n");
    fprintf(stderr, "    0: ニアレストネイバー法\n");
    fprintf(stderr, "    1: バイリニア法\n");
    fprintf(stderr, "    2: バイキュービック法\n");
    fprintf(stderr, "    3: AI超解像 (OpenVINO) ※拡大率は 2.0, 3.0, 4.0 のいずれかを指定\n");
    fprintf(stderr, "  [シャープ化強度k]\n");
    fprintf(stderr, "    0.0 でシャープ化なし、1.0~2.5程度で輪郭強調\n");
    fprintf(stderr, "  [AIモデルパス]\n");
    fprintf(stderr, "    補間モード3を使う場合のみ必須。省略時は環境変数 %s を参照します。\n", AI_MODEL_ENV_VAR);
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        PrintUsage(argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *inputFile = argv[1];
    double scale = atof(argv[2]);
    int method = atoi(argv[3]);
    double k = atof(argv[4]);
    int filterSize = 3;

    if (scale <= 0.0)
    {
        fprintf(stderr, "エラー: 拡大率は0より大きな値を指定してください\n");
        exit(EXIT_FAILURE);
    }

    // --- 0. 処理前の安全性チェック (限界拡大率の計算) ---
    int imgWidth, imgHeight, channels;

    if (!stbi_info(inputFile, &imgWidth, &imgHeight, &channels))
    {
        fprintf(stderr, "エラー: 画像情報の取得に失敗しました: %s\n", inputFile);
        exit(EXIT_FAILURE);
    }

    double maxScaleMemory = sqrt((double)INT_MAX / ((double)imgWidth * imgHeight * 3.0));
    double maxScaleWidth = 65535.0 / (double)imgWidth;
    double maxScaleHeight = 65535.0 / (double)imgHeight;

    double maxScale = maxScaleMemory;
    if (maxScaleWidth < maxScale)
        maxScale = maxScaleWidth;
    if (maxScaleHeight < maxScale)
        maxScale = maxScaleHeight;

    printf("========================================\n");
    printf("入力画像サイズ: %d x %d ピクセル\n", imgWidth, imgHeight);
    printf("計算可能な最大拡大率: 約 %.2f 倍\n", maxScale);
    printf("要求された拡大率: %.2f 倍\n", scale);
    printf("========================================\n");

    if (scale > maxScale)
    {
        fprintf(stderr, "\n[エラー] 要求された拡大率は処理限界を超えています。\n");
        fprintf(stderr, "※ メモリ(2GB)の限界、またはJPEG規格(65535px)の限界を超過します。\n");
        fprintf(stderr, "安全に処理できるのは %.2f 倍までです。処理を中止します。\n", maxScale);
        exit(EXIT_FAILURE);
    }
    printf("\n[チェックOK] 要求された拡大率は処理可能です。処理を開始します...\n\n");

    // AI超解像モードの場合は、先にモデルパスの妥当性を確認しておく
    const char *modelPath = NULL;
    if (method == 3)
    {
        int intScale = (int)scale;
        if (intScale != 2 && intScale != 3 && intScale != 4)
        {
            fprintf(stderr, "\n[エラー] AI超解像の拡大率は 2.0, 3.0, 4.0 のいずれかの整数を指定してください。\n");
            exit(EXIT_FAILURE);
        }

        modelPath = ResolveModelPath(argc, argv);
        if (modelPath == NULL)
        {
            fprintf(stderr, "\n[エラー] AI超解像のモデルパスが指定されていません。\n");
            fprintf(stderr, "第5引数でモデルの.xmlパスを指定するか、環境変数 %s を設定してください。\n", AI_MODEL_ENV_VAR);
            exit(EXIT_FAILURE);
        }
    }

    char *outputFile = MakeOutputFileName(inputFile, "_Scaled_Sharpened.jpg");

    // 1. 元画像の読み込み
    unsigned char *inputImage = stbi_load(inputFile, &imgWidth, &imgHeight, &channels, 3);
    if (inputImage == NULL)
    {
        fprintf(stderr, "画像の読み込みに失敗しました: %s\n", inputFile);
        free(outputFile);
        exit(EXIT_FAILURE);
    }

    // 2. 拡大処理実行
    long outWidth, outHeight;
    unsigned char *scaledImage = NULL;

    if (method == 3)
    {
        int outW, outH;
        scaledImage = ScaleImageAI_OpenCV(inputImage, imgWidth, imgHeight, (int)scale, &outW, &outH, modelPath);

        if (scaledImage == NULL)
        {
            fprintf(stderr, "AI超解像の処理に失敗しました。\n");
            free(outputFile);
            stbi_image_free(inputImage);
            exit(EXIT_FAILURE);
        }
        outWidth = (long)outW;
        outHeight = (long)outH;
        printf("拡大後サイズ(AI): %ld x %ld\n", outWidth, outHeight);
    }
    else
    {
        // 従来の補間モードの場合
        scaledImage = ScaleImage(inputImage, imgWidth, imgHeight, scale, method, &outWidth, &outHeight);
        printf("拡大後サイズ: %ld x %ld\n", outWidth, outHeight);
    }

    // 3. シャープ化処理の適用
    unsigned char *sharpenedImage = SharpenImage(scaledImage, outWidth, outHeight, filterSize, k);

    // 4. 結果の出力
    if (stbi_write_jpg(outputFile, (int)outWidth, (int)outHeight, 3, sharpenedImage, 90))
    {
        printf("出力完了: %s\n", outputFile);
    }
    else
    {
        fprintf(stderr, "出力ファイルの保存に失敗しました\n");
    }

    // 5. メモリの解放
    free(outputFile);
    stbi_image_free(inputImage);

    if (sharpenedImage != scaledImage)
    {
        free(sharpenedImage);
    }
    free(scaledImage);

    return 0;
}