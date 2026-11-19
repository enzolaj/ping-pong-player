import rclpy
from rclpy.node import Node


class InterceptPlanner(Node):
    def __init__(self):
        super().__init__('intercept_planner')


def main(args=None):
    rclpy.init(args=args)
    node = InterceptPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
