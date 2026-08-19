import csv
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_PATH = "src/bagfiles/bag_span_all/"        # dossier du bag (pas le .db3)
TOPIC_NAME = "/span/odom"        # adapte si besoin
CSV_PATH = "src/shared/resource/map_bag.csv"

storage_options = StorageOptions(
    uri=BAG_PATH,
    storage_id="mcap"
)

converter_options = ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader = SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

if TOPIC_NAME not in type_map:
    raise RuntimeError(f"Topic {TOPIC_NAME} not found in bag")

msg_type = get_message(type_map[TOPIC_NAME])

points = []

i = 0

while reader.has_next():
    topic, data, t = reader.read_next()

    if topic != TOPIC_NAME:
        continue

    if i % 10 != 0:
        i += 1
        continue

    msg = deserialize_message(data, msg_type)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z

    points.append(("map",x, y, z))

    i += 1


print(f"Extracted {len(points)} points")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["segment", "E", "N", "U"])
    writer.writerows(points)

print(f"Saved to {CSV_PATH}")
