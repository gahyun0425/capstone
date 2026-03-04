#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _dtype_and_channels(encoding: str):
    """Map ROS Image encoding to (numpy dtype, channels). Minimal set."""
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
    """Convert sensor_msgs/Image to numpy array without cv_bridge."""
    dtype, ch = _dtype_and_channels(msg.encoding)
    # msg.step is bytes per row
    expected_row_bytes = msg.width * ch * np.dtype(dtype).itemsize
    if msg.step != expected_row_bytes:
        # handle padded rows (rare). We'll read row by row then reshape.
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
        if ch > 1:
            out = data.reshape((msg.height, msg.width, ch))
        else:
            out = data.reshape((msg.height, msg.width))

    # Endianness handling
    # msg.is_bigendian: 0 little, 1 big
    if msg.is_bigendian and out.dtype.itemsize > 1:
        out = out.byteswap().newbyteorder()

    return out


class ZedViewer(Node):
    def __init__(self, left_color: str, right_color: str | None, depth: str | None):
        super().__init__("zed_viewer")

        self.left_color_topic = left_color
        self.right_color_topic = right_color
        self.depth_topic = depth

        self.latest = {"lc": None, "rc": None, "d": None}

        self.sub_lc = self.create_subscription(Image, self.left_color_topic, self.cb_left_color, 10)
        self.sub_rc = None
        self.sub_d = None

        if self.right_color_topic:
            self.sub_rc = self.create_subscription(Image, self.right_color_topic, self.cb_right_color, 10)
        if self.depth_topic:
            self.sub_d = self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)

        self.get_logger().info(f"Subscribe zed left  : {self.left_color_topic}")
        if self.right_color_topic:
            self.get_logger().info(f"Subscribe zed right : {self.right_color_topic}")
        if self.depth_topic:
            self.get_logger().info(f"Subscribe zed depth : {self.depth_topic}")

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

    def cb_left_color(self, msg: Image):
        img = imgmsg_to_numpy(msg)  # expects rgb8 from your pipeline
        if msg.encoding.lower() == "rgb8":
            self.latest["lc"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() == "bgr8":
            self.latest["lc"] = img
        else:
            # best effort
            self.latest["lc"] = img[:, :, :3] if img.ndim == 3 else img

    def cb_right_color(self, msg: Image):
        img = imgmsg_to_numpy(msg)
        if msg.encoding.lower() == "rgb8":
            self.latest["rc"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() == "bgr8":
            self.latest["rc"] = img
        else:
            self.latest["rc"] = img[:, :, :3] if img.ndim == 3 else img

    def cb_depth(self, msg: Image):
        self.latest["d"] = imgmsg_to_numpy(msg)

    def on_timer(self):
        if self.latest["lc"] is not None:
            cv2.imshow("zed/left", self.latest["lc"])
        if self.latest["rc"] is not None:
            cv2.imshow("zed/right", self.latest["rc"])
        if self.latest["d"] is not None:
            cv2.imshow("zed/depth", self._depth_to_vis(self.latest["d"]))

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
    ap.add_argument("--left_color", default="/zed_mini/zed_node/left/image_rect_color")
    ap.add_argument("--right_color", default="/zed_mini/zed_node/right/image_rect_color")
    ap.add_argument("--depth", default="/zed_mini/zed_node/depth/depth_registered")
    ap.add_argument("--no_right", action="store_true", default=False)
    ap.add_argument("--no_depth", action="store_true", default=False)
    args = ap.parse_args()

    rclpy.init()
    node = ZedViewer(
        left_color=args.left_color,
        right_color=None if args.no_right else args.right_color,
        depth=None if args.no_depth else args.depth,
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()