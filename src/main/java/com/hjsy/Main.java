package com.hjsy;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
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
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        // 1. 加载预训练的ResNet50模型
        Criteria<Image, Classifications> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelName("resnet50")
                .optEngine("PyTorch")  // 或 "MXNet", 取决于您的环境
                .build();

        ZooModel<Image, Classifications> baseModel = ModelZoo.loadModel(criteria);

        // 2. 获取模型的基础结构
        SymbolBlock baseBlock = (SymbolBlock) baseModel.getBlock();

        // 3. 创建新模型
        Model model = Model.newInstance("cube-detector");

        // 4. 创建新的网络结构
        SequentialBlock newBlock = new SequentialBlock();

        // 5. 提取ResNet的特征提取部分（除了最后一层全连接层）
        // 这里需要根据实际的模型结构来调整
        // 假设ResNet50的结构是：[conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]

        // 获取ResNet50的最后一层全连接层之前的所有层
        baseBlock.removeLastBlock();  // 移除最后的fc层
        newBlock.add(baseBlock);

        // 6. 冻结特征提取部分的参数，使其在训练中不更新
        // 注意：在DJL中，可以通过设置参数的requiresGradient属性来实现
        baseBlock.freezeParameters(true);

        // 7. 添加新的分类层 - 假设我们有两个类别：立方体和非立方体
        // 获取特征提取器输出的特征维度，通常是2048（对于ResNet50）
        int featureDim = 2048;  // ResNet50的特征维度
        int numClasses = 2;     // 立方体和非立方体

        // 添加新的全连接层
        newBlock.add(Linear.builder()
                .setUnits(numClasses)
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
            // 这里需要您准备好的训练数据集和验证数据集
            // Dataset trainDataset = ...
            // Dataset validationDataset = ...

            System.out.println("开始训练模型...");
            int numEpochs = 10;

            // 实际训练代码（取决于您的数据集）
            /*
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                System.out.printf("Epoch %d/%d\n", epoch + 1, numEpochs);

                // 训练一个epoch
                for (Batch batch : trainer.iterateDataset(trainDataset)) {
                    EasyTrain.trainBatch(trainer, batch);
                    trainer.step();
                    batch.close();
                }

                // 验证
                for (Batch batch : trainer.iterateDataset(validationDataset)) {
                    EasyTrain.validateBatch(trainer, batch);
                    batch.close();
                }

                // 输出训练指标
                float accuracy = trainer.getEvaluators().get(0).getMetric().getValue();
                System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
            }
            */

            // 13. 保存模型
            model.save(Paths.get("./"), "cube-detector");
            System.out.println("模型已保存为 cube-detector");
        }

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
        savedModel.load(Paths.get("./"), "cube-detector");

        try (Predictor<Image, Classifications> predictor = savedModel.newPredictor(translator)) {
            // 16. 在这里可以使用predictor进行预测
            // Image img = ...
            // Classifications result = predictor.predict(img);
            // System.out.println(result);
        }
    }
}
