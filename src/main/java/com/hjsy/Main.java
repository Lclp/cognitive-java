package com.hjsy;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDList;
import ai.djl.training.EasyTrain;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.translate.Pipeline;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        // 1. 加载预训练的ResNet50模型
        Criteria<Image, Classifications> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelName("traced_resnet50")
                .optEngine("PyTorch")  // 或 "MXNet", 取决于您的环境
                .build();

        ZooModel<Image, Classifications> baseModel = ModelZoo.loadModel(criteria);

        // 2. 创建新模型
        Model model = Model.newInstance("cube-detector");

        // 3. 创建新的网络结构
        SequentialBlock newBlock = new SequentialBlock();

        // 4. 获取预训练模型的Block
        Block baseBlock = baseModel.getBlock();

        // 5. 添加预训练模型的Block到新模型
        newBlock.add(baseBlock);

        // 6. 冻结预训练模型的参数
        baseBlock.freezeParameters(true);

        // 7. 添加新的分类层
        // 这里我们使用一个适配器层来处理ResNet50的输出
        newBlock.add(ndList -> {
            // 假设ResNet50的输出是[batch_size, 2048, 1, 1]
            // 我们需要将其转换为[batch_size, 2048]
            return new NDList(ndList.get(0).squeeze());
        });

        // 添加新的全连接层，用于分类
        newBlock.add(Linear.builder()
                .setUnits(2)  // 2个类别：立方体和非立方体
                .build());

        // 8. 设置模型的网络结构
        model.setBlock(newBlock);

        // 9. 配置训练参数
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optOptimizer(
                        Optimizer.adam()
                                .build()
                );

        // 10. 创建训练器
        try (Trainer trainer = model.newTrainer(config)) {
            // 11. 初始化训练器
            trainer.initialize(new Shape(32, 3, 224, 224));

            // 12. 训练模型
            // 创建数据预处理管道
            Pipeline pipeline = new Pipeline()
                    .add(new Resize(224, 224))
                    .add(new ToTensor())
                    .add(new Normalize(
                            new float[] {0.485f, 0.456f, 0.406f},
                            new float[] {0.229f, 0.224f, 0.225f}));

            // 加载训练数据集
            ImageFolder trainDataset = ImageFolder.builder()
                    .setRepositoryPath(Paths.get("dataset")) // 替换为您的数据集路径
                    .optPipeline(pipeline)
                    .setSampling(32, true) // 批量大小为32
                    .build();

            trainDataset.prepare();

            // 可选：创建验证数据集
            ImageFolder validationDataset = ImageFolder.builder()
                    .setRepositoryPath(Paths.get("dataset_test")) // 如果有验证集
                    .optPipeline(pipeline)
                    .setSampling(32, true)
                    .build();

            validationDataset.prepare();

            System.out.println("开始训练模型...");
            int numEpochs = 10;

            // 实际训练代码
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                System.out.printf("Epoch %d/%d\n", epoch + 1, numEpochs);

                // 训练一个epoch
                int batchCount = 0;
                for (Batch batch : trainer.iterateDataset(trainDataset)) {
                    EasyTrain.trainBatch(trainer, batch);
                    trainer.step();
                    batch.close();

                    // 每处理10个批次打印一次进度
                    if (++batchCount % 10 == 0) {
                        System.out.printf("Processed %d batches\n", batchCount);
                    }
                }

                // 验证
                if (validationDataset != null) {
                    System.out.println("Validating...");
                    batchCount = 0;
                    for (Batch batch : trainer.iterateDataset(validationDataset)) {
                        EasyTrain.validateBatch(trainer, batch);
                        batch.close();

                        // 每处理10个批次打印一次进度
                        if (++batchCount % 10 == 0) {
                            System.out.printf("Validated %d batches\n", batchCount);
                        }
                    }

                    // 输出训练指标
                    System.out.println("验证结果:");
                    for (Evaluator evaluator : trainer.getEvaluators()) {
                        System.out.println(evaluator);
                    }
                }
            }
        }
        Path modelPath = Paths.get("./models/");
        model.save(modelPath, "cube-detector-01");

        // 14. 创建模型推理所需的翻译器
        List<String> classes = Arrays.asList("非立方体", "立方体");

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[] {0.485f, 0.456f, 0.406f},
                        new float[] {0.229f, 0.224f, 0.225f}))
                .optSynset(classes)
                .build();

        // 15. 加载保存的模型并创建预测器
        Model savedModel = Model.newInstance("cube-detector");
        savedModel.load(modelPath, "cube-detector-01");

        try (Predictor<Image, Classifications> predictor = savedModel.newPredictor(translator)) {
            // 16. 单个图像预测
            String testImagePath = "test.jpg"; // 替换为您的测试图像路径
            Image img = ImageFactory.getInstance().fromFile(Paths.get(testImagePath));

            // 进行预测
            Classifications result = predictor.predict(img);

            // 打印预测结果
            System.out.println("预测结果:");
            System.out.println(result);

            // 获取最可能的类别
            String topClassName = result.best().getClassName();
            double topProbability = result.best().getProbability();
            System.out.printf("图像最可能是: %s，概率: %.2f%%\n",
                    topClassName, topProbability * 100);

            // 批量预测多个图像
            //String testImagesDir = "path/to/test/images"; // 替换为您的测试图像文件夹路径
            //java.io.File dir = new java.io.File(testImagesDir);
            //java.io.File[] files = dir.listFiles((d, name) -> name.endsWith(".jpg") || name.endsWith(".png"));
            //
            //if (files != null) {
            //    System.out.println("\n批量预测结果:");
            //    for (java.io.File file : files) {
            //        Image testImg = ImageFactory.getInstance().fromFile(file.toPath());
            //        Classifications testResult = predictor.predict(testImg);
            //
            //        System.out.printf("图像 %s: 预测为 %s，概率: %.2f%%\n",
            //                file.getName(),
            //                testResult.best().getClassName(),
            //                testResult.best().getProbability() * 100);
            //    }
            //}
        }
    }
}
