import rclpy
from rclpy.node import Node


class TrajectoryEstimator(Node):
    def __init__(self):
        super().__init__('trajectory_estimator')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
