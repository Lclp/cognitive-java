package com.hjsy;

import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.util.ArrayList;
import java.util.List;
import static org.bytedeco.opencv.global.opencv_imgproc.findContours;
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_imgproc.Vec2fVector;
import org.bytedeco.opencv.opencv_imgproc.Vec4fVector;
import org.bytedeco.opencv.opencv_imgproc.Vec4iVector;

public class Main {

    // 不需要手动加载库，JavaCV 会自动处理
    // JavaCV 不需要静态初始化块来加载库

    public static boolean detectCube(String imagePath) {
        Mat originalImage = imread(imagePath);
        if (originalImage.empty()) {
            System.out.println("无法读取图像: " + imagePath);
            return false;
        }

        // 转换为灰度图
        Mat grayImage = new Mat();
        cvtColor(originalImage, grayImage, COLOR_BGR2GRAY);

        // 应用高斯模糊减少噪声
        GaussianBlur(grayImage, grayImage, new Size(5, 5), 0);

        // 自适应二值化 - 处理不均匀照明
        Mat binaryImage = new Mat();
        adaptiveThreshold(grayImage, binaryImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

        // 形态学闭操作，闭合线条间的小缝隙
        Mat closedImage = new Mat();
        Mat closeKernel = getStructuringElement(MORPH_RECT, new Size(5, 5));
        morphologyEx(binaryImage, closedImage, MORPH_CLOSE, closeKernel);

        // 边缘检测
        Mat edges = new Mat();
        Canny(grayImage, edges, 50, 150, 3, false);

        // 膨胀边缘，使线条更连贯 - 修复方法调用
        Mat dilatedEdges = new Mat();
        Mat dilateKernel = getStructuringElement(MORPH_RECT, new Size(3, 3));
        dilate(edges, dilatedEdges, dilateKernel);
        // 再次膨胀以增强效果
        dilate(dilatedEdges, dilatedEdges, dilateKernel);

        // 使用 Hough 变换检测直线，调整参数
        Vec4iVector lines = new Vec4iVector();
        HoughLinesP(dilatedEdges, lines, 1, Math.PI/180, 80, 50, 10);

        // 创建白色背景图像
        Mat resultImage = new Mat(originalImage.size(), originalImage.type(), new Scalar(255, 255, 255, 255));

        // 存储过滤后的线条
        List<int[]> filteredLines = new ArrayList<>();

        // 绘制检测到的直线
        for (long i = 0; i < lines.size(); i++) {
            int[] line = new int[4];
            lines.get(i).get(line);  // 将向量元素复制到数组中
            int x1 = line[0];
            int y1 = line[1];
            int x2 = line[2];
            int y2 = line[3];

            // 计算线条长度
            double length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));

            // 只保留较长的线条
            if (length > 30) {
                filteredLines.add(line);
                line(resultImage, new Point(x1, y1), new Point(x2, y2),
                        new Scalar(0, 255, 0, 255), 2, LINE_8, 0);
            }
        }

        // 保存结果图像
        imwrite("enhanced_hough_lines_result.jpg", resultImage);

        try {
            // 查找轮廓 - 使用完全限定名称调用方法
            MatVector contours = new MatVector();
            Mat hierarchy = new Mat();
            // 创建一个临时的二值图像用于查找轮廓
            Mat contourInput = dilatedEdges.clone();
            org.bytedeco.opencv.global.opencv_imgproc.findContours(
                    contourInput, contours, hierarchy,
                    org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL,
                    org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE
            );

            // 分析轮廓特征以检测立方体
            boolean isCube = analyzeCubeFeatures(contours, originalImage.size());
            return isCube;
        } catch (Exception e) {
            System.err.println("轮廓检测过程中出错: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }





    private static boolean analyzeCubeFeatures(MatVector contours, Size imageSize) {
        // 立方体特征计数器
        int lineCount = 0;        // 直线数量
        int cornerCount = 0;      // 角点数量
        int quadrilateralCount = 0; // 四边形数量

        // 最小轮廓面积阈值（过滤掉太小的轮廓）
        double minContourArea = imageSize.width() * imageSize.height() * 0.001;

        // 分析每个轮廓
        for (long i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            double contourArea = contourArea(contour);
            if (contourArea < minContourArea) continue;

            // 轮廓周长
            double perimeter = arcLength(contour, true);

            // 轮廓近似
            Mat approxCurve = new Mat();
            approxPolyDP(contour, approxCurve, 0.04 * perimeter, true);

            // 获取近似后的顶点数
            int vertices = approxCurve.rows();

            // 检测直线段
            if (vertices == 2) {
                lineCount++;
            }

            // 检测角点（通常是三个点的轮廓）
            if (vertices == 3) {
                cornerCount++;
            }

            // 检测四边形（可能是立方体的面）
            if (vertices == 4) {
                quadrilateralCount++;
            }
        }

        // 打印检测到的特征数量（调试用）
        System.out.println("直线数量: " + lineCount);
        System.out.println("角点数量: " + cornerCount);
        System.out.println("四边形数量: " + quadrilateralCount);

        // 立方体判断逻辑
        // 一个标准立方体通常有:
        // - 至少9条可见边（理想情况下是12条，但手绘可能不完整）
        // - 至少7个可见角点（理想情况下是8个）
        // - 至少2个四边形面（理想情况下是3个可见面）
        boolean isCube = (lineCount >= 9 || cornerCount >= 6 || quadrilateralCount >= 2);

        // 额外的启发式规则：检测平行线和垂直线
        boolean hasParallelAndPerpendicularLines = detectParallelAndPerpendicularLines(contours);

        // 综合判断
        return isCube && hasParallelAndPerpendicularLines;
    }

    private static boolean detectParallelAndPerpendicularLines(MatVector contours) {
        List<Line> lines = extractLines(contours);

        int parallelPairs = 0;
        int perpendicularPairs = 0;

        // 比较所有线对，检查平行和垂直关系
        for (int i = 0; i < lines.size(); i++) {
            for (int j = i + 1; j < lines.size(); j++) {
                Line line1 = lines.get(i);
                Line line2 = lines.get(j);

                // 计算两条线的角度差
                double angleDiff = Math.abs(line1.angle - line2.angle);

                // 平行线判断（角度差接近0度或180度）
                if (angleDiff < 10 || Math.abs(angleDiff - 180) < 10) {
                    parallelPairs++;
                }

                // 垂直线判断（角度差接近90度）
                if (Math.abs(angleDiff - 90) < 10) {
                    perpendicularPairs++;
                }
            }
        }

        System.out.println("平行线对数量: " + parallelPairs);
        System.out.println("垂直线对数量: " + perpendicularPairs);

        // 立方体应该至少有3对平行线和3对垂直线
        return parallelPairs >= 3 && perpendicularPairs >= 3;
    }

    private static List<Line> extractLines(MatVector contours) {
        List<Line> lines = new ArrayList<>();

        for (long i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            double perimeter = arcLength(contour, true);

            // 轮廓近似
            Mat approxCurve = new Mat();
            approxPolyDP(contour, approxCurve, 0.04 * perimeter, true);

            // 获取点数组
            int pointCount = approxCurve.rows();

            // 如果只有两个点，认为是一条线
            if (pointCount == 2) {
                Point point1 = new Point(approxCurve.ptr(0));
                Point point2 = new Point(approxCurve.ptr(1));

                double dx = point2.x() - point1.x();
                double dy = point2.y() - point1.y();
                double angle = Math.toDegrees(Math.atan2(dy, dx));

                // 角度范围标准化到0-180度
                if (angle < 0) angle += 180;

                lines.add(new Line(point1, point2, angle));
            }
            // 如果有多个点，分解为多条线
            else if (pointCount > 2) {
                for (int j = 0; j < pointCount; j++) {
                    Point p1 = new Point(approxCurve.ptr(j));
                    Point p2 = new Point(approxCurve.ptr((j + 1) % pointCount));

                    double dx = p2.x() - p1.x();
                    double dy = p2.y() - p1.y();
                    double angle = Math.toDegrees(Math.atan2(dy, dx));

                    // 角度范围标准化到0-180度
                    if (angle < 0) angle += 180;

                    lines.add(new Line(p1, p2, angle));
                }
            }
        }

        return lines;
    }

    // 辅助类：表示一条线段
    static class Line {
        Point start;
        Point end;
        double angle; // 线的角度（0-180度）

        public Line(Point start, Point end, double angle) {
            this.start = start;
            this.end = end;
            this.angle = angle;
        }
    }

    // 主函数示例
    public static void main(String[] args) {
        String imagePath = "test.jpg";
        boolean isCube = detectCube(imagePath);
        System.out.println("是否为立方体: " + isCube);
    }
}
