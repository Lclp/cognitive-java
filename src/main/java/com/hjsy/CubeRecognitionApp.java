package com.hjsy;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class CubeRecognitionApp {

    private static final String MODEL_DIR = "models";
    private static final String MODEL_NAME = "cube-detector"; // 模型名称
    private static final String DATASET_DIR = "dataset";
    private static final String[] CLASS_NAMES = {"cube", "not_cube"};
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;
    private static final int BATCH_SIZE = 10;
    private static final int EPOCHS = 30;
    private static final int NUM_CHANNELS = 3;  // RGB图像

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        // 确保目录存在
        Path modelDir = Paths.get(MODEL_DIR);
        Files.createDirectories(modelDir);

        // 检查模型是否已存在
        Path modelPath = Paths.get(MODEL_DIR);
        boolean modelExists = Files.exists(modelPath) && Files.list(modelPath)
                .anyMatch(path -> path.getFileName().toString().startsWith(MODEL_NAME));

        if (!modelExists) {
            System.out.println("=== 开始训练模型 ===");
            trainModel();
        } else {
            System.out.println("=== 模型已存在，跳过训练 ===");
        }

        System.out.println("\n=== 开始测试模型 ===");
        // 测试一些样本图片
        String testImageDir = DATASET_DIR + "_test";
        File cubeDir = new File(testImageDir + "/cube");
        File notCubeDir = new File(testImageDir + "/not_cube");

        // 加载模型
        Model model = loadModel();
        try {
            if (cubeDir.exists() && cubeDir.isDirectory()) {
                File[] cubeImages = cubeDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") ||
                        name.toLowerCase().endsWith(".jpeg") ||
                        name.toLowerCase().endsWith(".png"));
                if (cubeImages != null && cubeImages.length > 0) {
                    System.out.println("测试包含立方体的图片:");
                    for (int i = 0; i < Math.min(3, cubeImages.length); i++) {
                        predictImage(cubeImages[i].getPath(), model);
                    }
                }
            }

            if (notCubeDir.exists() && notCubeDir.isDirectory()) {
                File[] notCubeImages = notCubeDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") ||
                        name.toLowerCase().endsWith(".jpeg") ||
                        name.toLowerCase().endsWith(".png"));
                if (notCubeImages != null && notCubeImages.length > 0) {
                    System.out.println("\n测试不包含立方体的图片:");
                    for (int i = 0; i < Math.min(3, notCubeImages.length); i++) {
                        predictImage(notCubeImages[i].getPath(), model);
                    }
                }
            }
        } finally {
            // 确保模型被关闭
            if (model != null) {
                model.close();
            }
        }
        System.out.println("\n=== 模型训练和测试完成 ===");
    }

    private static void trainModel() throws IOException, ModelException, TranslateException {
        System.out.println("创建简单CNN模型...");

        try (Model model = Model.newInstance(MODEL_NAME)) {
            // 创建一个简单的CNN网络
            SequentialBlock block = new SequentialBlock();

            // 第一个卷积层
            block.add(Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .setFilters(16)
                    .build());
            block.add(Activation::relu);
            block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

            // 第二个卷积层
            block.add(Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .setFilters(32)
                    .build());
            block.add(Activation::relu);
            block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

            // 全连接层
            block.add(Blocks.batchFlattenBlock());
            block.add(Linear.builder().setUnits(64).build());
            block.add(Activation::relu);
            block.add(Linear.builder().setUnits(CLASS_NAMES.length).build());

            model.setBlock(block);

            // 加载训练和验证数据集
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);
            RandomAccessDataset validationSet = getDataset(Dataset.Usage.TEST);

            // 配置训练参数
            DefaultTrainingConfig config = setupTrainingConfig();

            try (Trainer trainer = model.newTrainer(config)) {
                // 初始化训练器 - 注意这里使用3通道
                trainer.initialize(new Shape(BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH));

                // 开始训练
                System.out.println("开始模型训练...");
                EasyTrain.fit(trainer, EPOCHS, trainingSet, validationSet);

                // 保存模型
                model.save(Paths.get(MODEL_DIR), MODEL_NAME);
                System.out.println("模型训练完成并保存到: " + Paths.get(MODEL_DIR).toAbsolutePath());
            }
        }
    }

    // 加载模型
    private static Model loadModel() throws IOException, ModelException {
        Path modelPath = Paths.get(MODEL_DIR);

        if (!Files.exists(modelPath)) {
            System.out.println("未找到模型目录。请先训练模型。");
            return null;
        }

        // 创建与训练时相同结构的模型 - 移除 try-with-resources
        Model model = Model.newInstance(MODEL_NAME);

        // 重新创建相同的网络结构
        SequentialBlock block = new SequentialBlock();

        // 第一个卷积层
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(16)
                .build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        // 第二个卷积层
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(32)
                .build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        // 全连接层
        block.add(Blocks.batchFlattenBlock());
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(CLASS_NAMES.length).build());

        model.setBlock(block);

        // 加载模型参数
        model.load(modelPath);

        return model;
    }

    private static void predictImage(String imagePath, Model model) throws IOException, ModelException, TranslateException {


        // 创建与训练时相同结构的模型
        try {

            // 创建图像分类转换器
            Pipeline pipeline = new Pipeline()
                    .add(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                    .add(new ToTensor())
                    .add(new Normalize(
                            new float[]{0.485f, 0.456f, 0.406f},
                            new float[]{0.229f, 0.224f, 0.225f}));

            ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                    .setPipeline(pipeline)
                    .optSynset(Arrays.asList(CLASS_NAMES))
                    .build();

            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                // 加载图像
                Image img = ImageFactory.getInstance().fromFile(Paths.get(imagePath));

                // 预测
                Classifications result = predictor.predict(img);

                // 输出结果
                System.out.println("图片: " + imagePath);
                System.out.println("预测结果: " + result);
            }
        } catch (Exception e) {
            System.out.println("预测错误: " + e.getMessage());
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig() {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optOptimizer(Adam.builder().optLearningRateTracker(
                        Tracker.fixed(0.001f)).build())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage) {
        // 使用ImageFolder加载数据集
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(DATASET_DIR))
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[]{0.485f, 0.456f, 0.406f},  // RGB通道的均值
                        new float[]{0.229f, 0.224f, 0.225f})) // RGB通道的标准差
                .setSampling(BATCH_SIZE, true)
                .build();

        try {
            dataset.prepare();
        } catch (IOException | TranslateException e) {
            throw new RuntimeException("Failed to prepare dataset", e);
        }

        return dataset;
    }
}
