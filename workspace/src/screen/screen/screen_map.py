import numpy as np


import rclpy
from rclpy.node import Node

from geometry_msgs.msg import (
    PolygonStamped,
    Point32,
)

from shared.resource.map import (
    map_ligne_droite,
    map_rond_point_haut,
    map_rond_point_bas,
    cheat_map,
    map_bag,
)


class MapDisplayNode(Node):
    def __init__(self):
        super().__init__('map_display_node')

        self.pub_map_ligne_droite = self.create_publisher(PolygonStamped, "display/map/ligne_droite", 10)
        self.pub_map_rond_point_haut = self.create_publisher(PolygonStamped, "display/map/rond_point_haut", 10)
        self.pub_map_rond_point_bas = self.create_publisher(PolygonStamped, "display/map/rond_point_bas", 10)
        self.pub_cheat_map = self.create_publisher(PolygonStamped, "display/map/smooth", 10)
        self.pub_map_bag = self.create_publisher(PolygonStamped, "display/map/bag", 10)

        self.display_maps()

    def display_maps(self):
        maps = [map_ligne_droite, map_rond_point_haut, map_rond_point_bas, cheat_map, map_bag]
        pubs = [self.pub_map_ligne_droite, self.pub_map_rond_point_haut, self.pub_map_rond_point_bas, self.pub_cheat_map, self.pub_map_bag]

        for map_data, pub in zip(maps, pubs):
            msg_map = PolygonStamped()
            msg_map.polygon.points = []
            for point in map_data.T:
                p = Point32()
                p.x = point[0]
                p.y = point[1]
                p.z = 0.0
                msg_map.polygon.points.append(p)
            msg_map.header.stamp = self.get_clock().now().to_msg()
            msg_map.header.frame_id = "world"
            pub.publish(msg_map)


def main(args=None):
    rclpy.init(args=args)
    node = MapDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass