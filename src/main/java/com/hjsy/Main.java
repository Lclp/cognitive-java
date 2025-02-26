package com.hjsy;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.ndarray.types.Shape;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        // Create image processor
        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .optApplyNormalization()
                .build();

        // Configure and download ResNet50 model
        Criteria<Image, float[]> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, float[].class)
                .optModelUrls("djl://ai.djl.pytorch/resnet50")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        System.out.println("Starting model download and loading...");
        try (ZooModel<Image, float[]> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, float[]> predictor = model.newPredictor()) {
                Path imageFile = Paths.get("test.jpg");
                if (!imageFile.toFile().exists()) {
                    System.out.println("Please make sure test.jpg exists in the project root directory");
                    return;
                }

                System.out.println("Loading image...");
                Image img = ImageFactory.getInstance().fromFile(imageFile);

                System.out.println("Running prediction...");
                float[] prediction = predictor.predict(img);

                System.out.println("Prediction results:");
                for (int i = 0; i < prediction.length; i++) {
                    if (prediction[i] > 0.01) {
                        System.out.printf("Class %d: %.2f%%\n", i, prediction[i] * 100);
                    }
                }
            }
        }
    }
}
