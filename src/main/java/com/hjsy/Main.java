package com.hjsy;

import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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

    public static boolean enhancedCubeDetection(String imagePath) {
        // 读取图像
        Mat originalImage = imread(imagePath);
        if (originalImage.empty()) {
            System.out.println("无法读取图像: " + imagePath);
            return false;
        }

        // 1. 优化的预处理
        Mat preprocessedImage = optimizedPreprocessing(originalImage);
        imwrite("preprocessed.jpg", preprocessedImage);

        // 2. 增强的边缘检测
        Mat edgeImage = enhancedEdgeDetection(preprocessedImage);
        imwrite("enhanced_edges.jpg", edgeImage);

        // 3. 改进的线条检测
        List<int[]> detectedLines = improvedLineDetection(edgeImage, originalImage);

        // 4. 立方体结构分析
        boolean isCube = detectCubeStructure(detectedLines, originalImage.size());

        System.out.println("立方体检测结果: " + (isCube ? "是立方体" : "不是立方体"));
        return isCube;
    }

    public static Mat optimizedPreprocessing(Mat originalImage) {
        // 转换为灰度图
        Mat grayImage = new Mat();
        cvtColor(originalImage, grayImage, COLOR_BGR2GRAY);

        // 对于单色图像，使用自适应阈值处理，保留更多细节
        Mat binaryImage = new Mat();
        // 减小块大小(从11到7)，提高局部对比度敏感度
        adaptiveThreshold(grayImage, binaryImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                THRESH_BINARY_INV, 7, 2);

        // 使用更小的核进行形态学操作，保留细节
        Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, new Size(2, 2));
        Mat processedImage = new Mat();
        // 先开操作去除噪点
        morphologyEx(binaryImage, processedImage, MORPH_OPEN, morphKernel);
        // 再闭操作连接断线
        morphologyEx(processedImage, processedImage, MORPH_CLOSE, morphKernel);

        return processedImage;
    }

    public static Mat enhancedEdgeDetection(Mat preprocessedImage) {
        // 使用更低的阈值进行Canny边缘检测
        Mat edges = new Mat();
        Canny(preprocessedImage, edges, 30, 90); // 降低阈值，捕获更多边缘

        // 轻微膨胀，连接断开的边缘，但不过度模糊
        Mat dilateKernel = getStructuringElement(MORPH_RECT, new Size(2, 2));
        Mat dilatedEdges = new Mat();
        dilate(edges, dilatedEdges, dilateKernel);

        return dilatedEdges;
    }

    public static List<int[]> improvedLineDetection(Mat edgeImage, Mat originalImage) {
        // 使用概率Hough变换检测线条，调整参数以适应手绘图像
        Vec4iVector lines = new Vec4iVector();
        // 降低阈值(从80到40)，减小最小线长(从50到25)，增大最大间隙(从10到15)
        HoughLinesP(edgeImage, lines, 1, Math.PI / 180, 40, 25, 15);

        // 存储检测到的线条
        List<int[]> detectedLines = new ArrayList<>();

        // 创建结果图像用于可视化
        Mat resultImage = originalImage.clone();

        // 处理检测到的线条
        for (long i = 0; i < lines.size(); i++) {
            int[] line = new int[4];
            lines.get(i).get(line);
            detectedLines.add(line);

            // 在结果图像上绘制线条 - 修复 line 方法参数
            line(resultImage, new Point(line[0], line[1]), new Point(line[2], line[3]),
                    new Scalar(0.0, 255.0, 0.0, 255.0), 2, LINE_8, 0);
        }

        // 保存结果图像
        imwrite("improved_line_detection.jpg", resultImage);

        return detectedLines;
    }



    public static boolean detectCubeStructure(List<int[]> lines, Size imageSize) {
        // 如果检测到的线条太少，可能不是立方体
        if (lines.size() < 9) { // 立方体至少需要9条可见边
            return false;
        }

        // 分析线条之间的几何关系
        // 1. 找出水平和垂直线条
        List<int[]> horizontalLines = new ArrayList<>();
        List<int[]> verticalLines = new ArrayList<>();

        for (int[] line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            double angle = Math.abs(Math.toDegrees(Math.atan2(y2 - y1, x2 - x1)));

            // 允许一定的误差范围，手绘图像线条不会完全水平或垂直
            if ((angle < 30 || angle > 150)) {
                horizontalLines.add(line);
            } else if ((angle > 60 && angle < 120)) {
                verticalLines.add(line);
            }
        }

        // 2. 检查透视线 - 既不是水平也不是垂直的线
        List<int[]> perspectiveLines = new ArrayList<>();
        for (int[] line : lines) {
            if (!horizontalLines.contains(line) && !verticalLines.contains(line)) {
                perspectiveLines.add(line);
            }
        }

        // 3. 验证立方体结构 - 至少需要有一定数量的各类线条
        boolean hasEnoughHorizontal = horizontalLines.size() >= 4;
        boolean hasEnoughVertical = verticalLines.size() >= 4;
        boolean hasEnoughPerspective = perspectiveLines.size() >= 1;

        // 4. 检查交点 - 立方体应该有多个线条交点
        int intersectionCount = countIntersections(lines);
        boolean hasEnoughIntersections = intersectionCount >= 8; // 立方体至少有8个顶点

        return hasEnoughHorizontal && hasEnoughVertical &&
                hasEnoughPerspective && hasEnoughIntersections;
    }

    // 计算线条交点数量
    private static int countIntersections(List<int[]> lines) {
        Set<Point> intersections = new HashSet<>();

        // 检查每对线条是否相交
        for (int i = 0; i < lines.size(); i++) {
            for (int j = i + 1; j < lines.size(); j++) {
                Point intersection = lineIntersection(lines.get(i), lines.get(j));
                if (intersection != null) {
                    // 使用一定的容差将接近的交点合并
                    boolean isNew = true;
                    for (Point existing : intersections) {
                        if (distance(existing, intersection) < 10) { // 10像素容差
                            isNew = false;
                            break;
                        }
                    }
                    if (isNew) {
                        intersections.add(intersection);
                    }
                }
            }
        }

        return intersections.size();
    }

    // 计算两条线的交点
    private static Point lineIntersection(int[] line1, int[] line2) {
        int x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
        int x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];

        // 线段方程: (x1,y1)+(x2-x1,y2-y1)*t1 = (x3,y3)+(x4-x3,y4-y3)*t2
        double denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);

        // 如果线段平行，则无交点
        if (Math.abs(denominator) < 1e-6) {
            return null;
        }

        double ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator;
        double ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator;

        // 如果交点在两条线段上
        if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
            int x = (int) (x1 + ua * (x2 - x1));
            int y = (int) (y1 + ua * (y2 - y1));
            return new Point(x, y);
        }

        return null;
    }

    // 计算两点之间的距离
    private static double distance(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p2.x() - p1.x(), 2) + Math.pow(p2.y() - p1.y(), 2));
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
        String imagePath = "111.png";
        boolean isCube = enhancedCubeDetection(imagePath);
        System.out.println("是否为立方体: " + isCube);
    }
}
