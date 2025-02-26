package com.hjsy;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Pipeline;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        // 创建预处理管道
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(224, 224))
                .add(new ToTensor())
                .add(new Normalize(
                        new float[] {0.485f, 0.456f, 0.406f},
                        new float[] {0.229f, 0.224f, 0.225f}));

        // 创建翻译器
        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .setPipeline(pipeline)
                .optSynsetArtifactName("synset.txt")
                .optApplySoftmax(true)
                .build();

        // 配置模型标准 - 使用更明确的模型规格
        Criteria<Image, Classifications> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelName("traced_resnet50") // 使用明确的模型名称
                .optFilter("layers", "50")
                .optFilter("dataset", "imagenet")
                .optEngine("PyTorch")
                .optProgress(new ProgressBar())
                .optTranslator(translator)
                .build();

        System.out.println("Starting model download and loading...");
        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
                Path imageFile = Paths.get("test.jpg");
                if (!imageFile.toFile().exists()) {
                    System.out.println("Please make sure test.jpg exists in the project root directory");
                    return;
                }

                System.out.println("Loading image...");
                Image img = ImageFactory.getInstance().fromFile(imageFile);

                System.out.println("Running prediction...");
                Classifications classifications = predictor.predict(img);

                // Print top-5 predictions
                System.out.println("\nTop 10 Predictions:");
                classifications.items().stream()
                        .limit(10)
                        .forEach(classification ->
                                System.out.printf("Class: %-30s Probability: %.2f%%\n",
                                        classification.getClassName(),
                                        classification.getProbability() * 100));
                var topPrediction = classifications.best();
                System.out.println("最佳预测: " + topPrediction.getClassName() +
                        " (置信度: " + topPrediction.getProbability() * 100 + "%)");
            }
        }
    }
}
