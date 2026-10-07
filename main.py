# CanMV K230 人脸检测 + LCD 实时显示
#
# 运行环境：CanMV K230 01Studio - 2G
# 功能：摄像头采集 -> 人脸检测模型推理 -> 在 LCD 屏幕上实时绘制人脸框
#
# 依赖（板子 /sdcard 上已自带）：
#   /sdcard/examples/kmodel/face_detection_320.kmodel
#   /sdcard/examples/utils/prior_data_320.bin

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *
import gc
import time
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo


class FaceDetectionApp(AIBase):
    """自定义人脸检测类，继承自 AIBase 基类"""

    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.5, nms_threshold=0.2,
                 rgb888p_size=[1280, 720], display_size=[800, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path                       # 模型文件路径
        self.model_input_size = model_input_size             # 模型输入分辨率
        self.confidence_threshold = confidence_threshold     # 置信度阈值
        self.nms_threshold = nms_threshold                   # NMS 阈值
        self.anchors = anchors                               # 锚点数据
        # sensor 给到 AI 的图像分辨率，宽度做 16 对齐
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        # 显示分辨率，宽度做 16 对齐
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)                         # 模型预处理
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                 nn.ai2d_format.NCHW_FMT,
                                 np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        """配置预处理：pad + resize"""
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right, _ = letterbox_pad_param(self.rgb888p_size,
                                                              self.model_input_size)
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        """后处理：使用 aidemo 提供的人脸检测后处理接口"""
        with ScopedTiming("postprocess", self.debug_mode > 0):
            post_ret = aidemo.face_det_post_process(self.confidence_threshold,
                                                    self.nms_threshold,
                                                    self.model_input_size[1],
                                                    self.anchors,
                                                    self.rgb888p_size,
                                                    results)
            if len(post_ret) == 0:
                return post_ret
            return post_ret[0]

    def draw_result(self, pl, dets):
        """把检测到的人脸框绘制到 OSD 图层上"""
        with ScopedTiming("display_draw", self.debug_mode > 0):
            pl.osd_img.clear()
            if dets:
                for det in dets:
                    # 检测框坐标从 AI 分辨率映射到显示分辨率
                    x, y, w, h = map(lambda v: int(round(v, 0)), det[:4])
                    x = x * self.display_size[0] // self.rgb888p_size[0]
                    y = y * self.display_size[1] // self.rgb888p_size[1]
                    w = w * self.display_size[0] // self.rgb888p_size[0]
                    h = h * self.display_size[1] // self.rgb888p_size[1]
                    # 画黄色矩形框（注意：颜色只能用 3 元组 RGB，带 alpha 的 4 元组不生效）
                    pl.osd_img.draw_rectangle(x, y, w, h,
                                              color=(255, 255, 0), thickness=3)
                    # 画置信度分数
                    if len(det) > 4:
                        conf = det[4]
                        pl.osd_img.draw_string_advanced(
                            x, max(y - 22, 0), 22, "%.2f" % conf,
                            color=(255, 0, 0))

    def draw_fps(self, pl, fps):
        """把实时帧率直接叠加显示到屏幕左上角（不画背景，只有文字遮挡画面）"""
        with ScopedTiming("display_fps", self.debug_mode > 0):
            text = "FPS: %.1f" % fps
            # 只画文字：OSD 图层由 clear() 清成透明，未绘制的像素透出视频画面。
            # 注意：不要画填充矩形做底色，那会遮挡画面；
            # 且颜色必须用 3 元组 RGB，带 alpha 的 4 元组在 OSD 上不生效。
            pl.osd_img.draw_string_advanced(10, 10, 28, text, color=(0, 255, 0))


if __name__ == "__main__":
    # 显示模式："lcd" 使用 LCD 屏幕；也可用 "auto" 自动识别
    display_mode = "lcd"
    # None 表示使用 SDK 默认屏幕分辨率
    display_size = None
    # 传给 AI 的图像分辨率
    rgb888p_size = [1280, 720]

    # 模型与参数
    kmodel_path = "/sdcard/examples/kmodel/face_detection_320.kmodel"
    confidence_threshold = 0.5
    nms_threshold = 0.2
    anchor_len = 4200
    det_dim = 4
    anchors_path = "/sdcard/examples/utils/prior_data_320.bin"
    anchors = np.fromfile(anchors_path, dtype=np.float).reshape((anchor_len, det_dim))

    # 初始化 PipeLine：负责摄像头采集、OSD 叠加和屏幕显示
    pl = PipeLine(rgb888p_size=rgb888p_size,
                  display_mode=display_mode,
                  display_size=display_size)
    pl.create()
    display_size = pl.get_display_size()

    # 初始化人脸检测实例
    face_det = FaceDetectionApp(kmodel_path,
                                model_input_size=[320, 320],
                                anchors=anchors,
                                confidence_threshold=confidence_threshold,
                                nms_threshold=nms_threshold,
                                rgb888p_size=rgb888p_size,
                                display_size=display_size,
                                debug_mode=0)
    face_det.config_preprocess()

    # 帧率统计器：tick() 每帧记录一次，fps() 读取当前帧率
    fps_clock = time.clock()

    try:
        while True:
            fps_clock.tick()                # 标记一帧开始
            img = pl.get_frame()            # 获取摄像头当前帧
            res = face_det.run(img)         # 推理
            face_det.draw_result(pl, res)   # 绘制人脸框
            face_det.draw_fps(pl, fps_clock.fps())  # 绘制实时帧率到 LCD
            pl.show_image()                 # 显示到 LCD
            gc.collect()                    # 垃圾回收
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        # 释放资源，避免重复初始化报 "sensor is already inited"
        face_det.deinit()
        pl.destroy()
