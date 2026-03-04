#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _dtype_and_channels(encoding: str):
    enc = encoding.lower()
    if enc in ("rgb8", "bgr8"):
        return np.uint8, 3
    if enc in ("rgba8", "bgra8"):
        return np.uint8, 4
    if enc in ("mono8", "8uc1"):
        return np.uint8, 1
    if enc in ("mono16", "16uc1"):
        return np.uint16, 1
    if enc in ("32fc1",):
        return np.float32, 1
    raise ValueError(f"Unsupported encoding: {encoding}")


def imgmsg_to_numpy(msg: Image):
    dtype, ch = _dtype_and_channels(msg.encoding)
    expected_row_bytes = msg.width * ch * np.dtype(dtype).itemsize

    if msg.step != expected_row_bytes:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        img = np.zeros((msg.height, msg.width, ch), dtype=dtype) if ch > 1 else np.zeros((msg.height, msg.width), dtype=dtype)
        row_bytes = msg.step
        for r in range(msg.height):
            row = data[r * row_bytes : r * row_bytes + expected_row_bytes]
            row = row.view(dtype)
            if ch > 1:
                img[r, :, :] = row.reshape(msg.width, ch)
            else:
                img[r, :] = row.reshape(msg.width)
        out = img
    else:
        data = np.frombuffer(msg.data, dtype=dtype)
        out = data.reshape((msg.height, msg.width, ch)) if ch > 1 else data.reshape((msg.height, msg.width))

    if msg.is_bigendian and out.dtype.itemsize > 1:
        out = out.byteswap().newbyteorder()

    return out


class MultiImageViewer(Node):
    def __init__(self, left_color, right_color, left_depth=None, right_depth=None):
        super().__init__("realsense_viewer")

        self.left_color_topic = left_color
        self.right_color_topic = right_color
        self.left_depth_topic = left_depth
        self.right_depth_topic = right_depth

        self.sub_lc = self.create_subscription(Image, self.left_color_topic, self.cb_left_color, 10)
        self.sub_rc = self.create_subscription(Image, self.right_color_topic, self.cb_right_color, 10)

        self.sub_ld = None
        self.sub_rd = None
        if self.left_depth_topic:
            self.sub_ld = self.create_subscription(Image, self.left_depth_topic, self.cb_left_depth, 10)
        if self.right_depth_topic:
            self.sub_rd = self.create_subscription(Image, self.right_depth_topic, self.cb_right_depth, 10)

        self.latest = {"lc": None, "rc": None, "ld": None, "rd": None}

        self.get_logger().info(f"Subscribe left  color: {self.left_color_topic}")
        self.get_logger().info(f"Subscribe right color: {self.right_color_topic}")
        if self.left_depth_topic:
            self.get_logger().info(f"Subscribe left  depth: {self.left_depth_topic}")
        if self.right_depth_topic:
            self.get_logger().info(f"Subscribe right depth: {self.right_depth_topic}")

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

    def cb_left_color(self, msg: Image):
        img = imgmsg_to_numpy(msg)
        # RealSense는 bgr8이 흔함. rgb8이면 BGR로 바꿔서 표시.
        if msg.encoding.lower() == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.latest["lc"] = img

    def cb_right_color(self, msg: Image):
        img = imgmsg_to_numpy(msg)
        if msg.encoding.lower() == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.latest["rc"] = img

    def cb_left_depth(self, msg: Image):
        self.latest["ld"] = imgmsg_to_numpy(msg)

    def cb_right_depth(self, msg: Image):
        self.latest["rd"] = imgmsg_to_numpy(msg)

    def on_timer(self):
        if self.latest["lc"] is not None:
            cv2.imshow("realsense_left/color", self.latest["lc"])
        if self.latest["rc"] is not None:
            cv2.imshow("realsense_right/color", self.latest["rc"])

        if self.latest["ld"] is not None:
            cv2.imshow("realsense_left/depth", self._depth_to_vis(self.latest["ld"]))
        if self.latest["rd"] is not None:
            cv2.imshow("realsense_right/depth", self._depth_to_vis(self.latest["rd"]))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            rclpy.shutdown()

    @staticmethod
    def _depth_to_vis(depth):
        d = depth.astype("float32")
        valid = d[d > 0]
        if valid.size == 0:
            return (d * 0).astype("uint8")
        lo, hi = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
        hi = max(hi, lo + 1e-6)
        d = (d - lo) / (hi - lo)
        d = (d * 255.0).clip(0, 255).astype("uint8")
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_color", default="/camera_l/color/image_raw")
    ap.add_argument("--right_color", default="/camera_r/color/image_raw")
    ap.add_argument("--left_depth", default="/camera_l/depth/image_raw")
    ap.add_argument("--right_depth", default="/camera_r/depth/image_raw")
    args = ap.parse_args()

    rclpy.init()
    node = MultiImageViewer(
        left_color=args.left_color,
        right_color=args.right_color,
        left_depth=args.left_depth,
        right_depth=args.right_depth,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()