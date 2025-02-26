package com.hjsy;

import ai.djl.Application;
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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
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
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class CubeRecognitionApp {

    private static final String MODEL_DIR = "models";
    private static final String DATASET_DIR = "dataset";
    private static final String[] CLASS_NAMES = {"cube", "not_cube"};
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        // 确保目录存在
        Path modelDir = Paths.get(MODEL_DIR);
        Files.createDirectories(modelDir);
        trainModel();
        predictImage(args[1]);

        // 检查命令行参数

    }

    private static void trainModel() throws IOException, ModelException, TranslateException {
        System.out.println("Loading DoodleNet base model...");

        // 加载预训练的DoodleNet模型
        Criteria<Image, Classifications> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelName("doodlenet")
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<Image, Classifications> doodleNet = ModelZoo.loadModel(criteria);
             Model model = Model.newInstance("cube-detector")) {

            // 获取DoodleNet的基础网络
            Block baseBlock = doodleNet.getBlock();

            // 创建新模型，替换最后一层以适应我们的分类任务
            SequentialBlock newBlock = modifyLastLayer(baseBlock, CLASS_NAMES.length);
            model.setBlock(newBlock);

            // 加载训练和验证数据集
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);
            RandomAccessDataset validationSet = getDataset(Dataset.Usage.TEST);

            // 配置训练参数
            DefaultTrainingConfig config = setupTrainingConfig();

            try (Trainer trainer = model.newTrainer(config)) {
                // 初始化训练器 - 使用DJL的Shape类
                trainer.initialize(new Shape(BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH));

                // 开始训练
                System.out.println("Starting model training...");
                EasyTrain.fit(trainer, EPOCHS, trainingSet, validationSet);

                // 保存模型
                Path modelPath = Paths.get(MODEL_DIR, "cube-model");
                model.save(modelPath, "cube-detector");
                System.out.println("Model trained and saved to: " + modelPath);
            }
        }
    }

    private static void predictImage(String imagePath) throws IOException, ModelException, TranslateException {
        Path modelPath = Paths.get(MODEL_DIR, "cube-model");

        if (!Files.exists(modelPath)) {
            System.out.println("Model not found. Please train the model first.");
            return;
        }

        // 加载我们的自定义模型
        Model model = Model.newInstance("cube-detector");
        model.load(modelPath);

        // 创建图像分类转换器
        Pipeline pipeline = new Pipeline()
                .add(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .add(new ToTensor())
                .add(new Normalize(new float[] {0.5f}, new float[] {0.5f}));

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .setPipeline(pipeline)
                .optSynset(Arrays.asList(CLASS_NAMES))
                .build();

        try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
            // 加载图像
            Image img = ImageFactory.getInstance().fromFile(Paths.get(imagePath));

            // 预处理图像 - 转为灰度图
            img = convertToGrayscale(img);

            // 预测
            Classifications result = predictor.predict(img);

            // 输出结果
            System.out.println("Prediction results:");
            System.out.println(result);
        }
    }

    private static SequentialBlock modifyLastLayer(Block baseBlock, int numClasses) {
        // 这里需要根据DoodleNet的具体结构修改
        SequentialBlock newBlock = new SequentialBlock();

        // 处理SequentialBlock类型的baseBlock
        if (baseBlock instanceof SequentialBlock) {
            SequentialBlock sequential = (SequentialBlock) baseBlock;
            // 获取子层列表
            var children = sequential.getChildren();
            // 复制除了最后一层以外的所有层
            for (int i = 0; i < children.size() - 1; i++) {
                // 从Pair中获取Block对象
                Block childBlock = children.get(i).getValue();
                newBlock.add(childBlock);
            }
        } else {
            // 如果不是SequentialBlock，可能需要更复杂的处理
            throw new UnsupportedOperationException("Base model block is not a SequentialBlock");
        }

        // 添加新的分类层
        newBlock.add(Linear.builder().setUnits(numClasses).build());

        return newBlock;
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
                //.setUsage(usage)
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(new float[] {0.5f}, new float[] {0.5f}))
                .setSampling(BATCH_SIZE, true)
                .build();

        try {
            dataset.prepare();
        } catch (IOException e) {
            throw new RuntimeException("Failed to prepare dataset", e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }

        return dataset;
    }

    private static Image convertToGrayscale(Image image) {
        // 使用NDArray操作将彩色图像转换为灰度图
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = image.toNDArray(manager);

            // 如果是彩色图像，转换为灰度图
            if (array.getShape().dimension() > 2 && array.getShape().get(2) == 3) {
                // RGB转灰度公式: 0.299 * R + 0.587 * G + 0.114 * B
                NDArray r = array.get(":, :, 0");
                NDArray g = array.get(":, :, 1");
                NDArray b = array.get(":, :, 2");

                NDArray gray = r.mul(0.299f).add(g.mul(0.587f)).add(b.mul(0.114f));
                gray = gray.expandDims(2); // 添加通道维度

                // 创建新图像
                return ImageFactory.getInstance().fromNDArray(gray);
            }

            // 如果已经是灰度图，直接返回
            return image;
        }
    }
}
