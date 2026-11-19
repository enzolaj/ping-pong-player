import rclpy
from rclpy.node import Node


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')


def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
