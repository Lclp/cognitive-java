package com.hjsy;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import okhttp3.*;

public class OpenAIVisionAPI {

    private static final String API_URL = "https://4.0.wokaai.com/v1/chat/completions";

    private static final String API_KEY = "sk-JdYZntmPsyPRP5qmWtGqGJuBHqb712gisSD8NcZCoI63KDOl"; // 替换为你的 中转 API 密钥

    public static void main(String[] args) {
        // 读取图片并转换为 Base64
        //String base64Image = encodeImageToBase64("./1231.jpg"); // 替换为图片路径
        //String base64Image = encodeImageToBase64("./123.jpg"); // 替换为图片路径
        //String base64Image = encodeImageToBase64("./112.png"); // 替换为图片路径
        //String base64Image = encodeImageToBase64("./114.png"); // 替换为图片路径
        //String base64Image = encodeImageToBase64("./no.png"); // 替换为图片路径
        String base64Image = encodeImageToBase64("./yes.jpg"); // 替换为图片路径


        if (base64Image != null) {
            // 调用 OpenAI Vision API
            try {
                String response = callOpenAIVisionAPI(base64Image);
                System.out.println("API 返回结果: " + response);
            } catch (IOException e) {
                System.err.println("调用 OpenAI API 出错: " + e.getMessage());
            }
        }
    }

    /**
     * 将图片文件转换为 Base64 编码字符串
     */
    private static String encodeImageToBase64(String imagePath) {
        try {
            byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));
            return java.util.Base64.getEncoder().encodeToString(imageBytes);
        } catch (IOException e) {
            System.err.println("读取图片文件失败: " + e.getMessage());
            return null;
        }
    }

    /**
     * 调用 OpenAI Vision API
     */
    private static String callOpenAIVisionAPI(String base64Image) throws IOException {
        OkHttpClient client = new OkHttpClient();

        // 构造请求体
        Map<String, Object> urlContent = new HashMap<>();
        urlContent.put("url", "data:image/jpeg;base64," + base64Image);

        Map<String, Object> imageContent = new HashMap<>();
        imageContent.put("type", "image_url");
        imageContent.put("image_url", urlContent);

        Map<String, Object> textContent = new HashMap<>();
        textContent.put("type", "text");
        textContent.put("text", "标准：\n" +
                "-如果图片中的物体8个顶点和12条边，表现出三维透视效果，则将其识别为正方体。\n" +
                "-如果图片中的物体不满足上述条件，则将其识别为非正方体。\n" +
                "-result为结果，枚举Y/N；score为评分，0-100整数；reason为结果的解释\n" +
                "返回json");

        List<Map<String, Object>> contentList = Arrays.asList(textContent, imageContent);

        Map<String, Object> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", contentList);

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", "gpt-4o"); // 替换为实际的模型名称
        requestBody.put("messages", Collections.singletonList(userMessage));
        requestBody.put("max_tokens", 300);

        ObjectMapper objectMapper = new ObjectMapper();
        String jsonBody = objectMapper.writeValueAsString(requestBody);


        // 创建请求
        Request request = new Request.Builder()
                .url(API_URL)
                .post(RequestBody.create(jsonBody, MediaType.parse("application/json")))
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .build();

        // 执行请求
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Unexpected code " + response);
            }
            // 使用 Jackson 解析主 JSON
            ObjectMapper respMapper = new ObjectMapper();
            JsonNode rootNode = respMapper.readTree(response.body().string());

            // 定位到 "choices" 数组中的 "content" 字段
            String content = rootNode
                    .path("choices")
                    .get(0)
                    .path("message")
                    .path("content")
                    .asText();

            // 提取 content 中的嵌套 JSON
            String extractedJson = extractJsonFromContent(content);

            // 解析嵌套 JSON
            JsonNode nestedJson = respMapper.readTree(extractedJson);

            // 输出解析结果
            System.out.println("Result: " + nestedJson.get("result").asText());
            System.out.println("Score: " + nestedJson.get("score").asInt());
            System.out.println("Reason: " + nestedJson.get("reason").asText());
            return "";
        }
    }

    /**
     * 从 content 字符串中提取 JSON 部分
     */
    private static String extractJsonFromContent(String content) {
        // 去掉 ```json 和 ``` 标记
        return content.replace("```json", "").replace("```", "").trim();
    }
}
