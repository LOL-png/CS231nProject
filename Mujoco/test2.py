from model import RobotArmAllegro

# Create robot instance
robot = RobotArmAllegro()

# Create visualization suite
viz_suite = RobotArmVisualizationSuite(robot)

# Run individual tests
viz_suite.test_pickup_task_visualization()

# Or run all tests
viz_suite.run_full_test_suite()