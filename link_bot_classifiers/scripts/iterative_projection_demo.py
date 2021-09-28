import tensorflow as tf

import rospy
from link_bot_classifiers.iterative_projection import iterative_projection, BaseProjectOpt
from visualization_msgs.msg import Marker


class ProjectOpt(BaseProjectOpt):
    def __init__(self):
        super().__init__(opt=tf.optimizers.Adam(0.05))

    def step(self, _, x_var: tf.Variable):
        variables = [x_var]
        with tf.GradientTape() as tape:
            loss = tf.square(x_var[1])
        gradients = tape.gradient(loss, variables)
        self.opt.apply_gradients(grads_and_vars=zip(gradients, variables))
        x_out = tf.convert_to_tensor(x_var)
        return x_out, False, []


def main():
    rospy.init_node('iterative_projection_demo')

    initial_value_pub = rospy.Publisher('initial_value', Marker, queue_size=10)
    x_pub = rospy.Publisher('x', Marker, queue_size=10)
    target_pub = rospy.Publisher('target', Marker, queue_size=10)

    def step_x_towards_target(target, x):
        return x + 0.05 * (target - x), []

    def viz_func(_, x, initial_value, target, __):
        s = 0.05
        initial_value_msg = Marker()
        initial_value_msg.header.frame_id = 'world'
        initial_value_msg.action = Marker.ADD
        initial_value_msg.type = Marker.SPHERE
        initial_value_msg.scale.x = s
        initial_value_msg.scale.y = s
        initial_value_msg.scale.z = s
        initial_value_msg.color.a = 1.0
        initial_value_msg.color.b = 1.0
        initial_value_msg.pose.position.x = initial_value[0]
        initial_value_msg.pose.position.y = 0
        initial_value_msg.pose.position.z = initial_value[1]
        initial_value_msg.pose.orientation.w = 1
        initial_value_pub.publish(initial_value_msg)

        x_msg = Marker()
        x_msg.header.frame_id = 'world'
        x_msg.action = Marker.ADD
        x_msg.type = Marker.SPHERE
        x_msg.scale.x = s
        x_msg.scale.y = s
        x_msg.scale.z = s
        x_msg.color.a = 1.0
        x_msg.color.r = 1.0
        x_msg.pose.position.x = x[0]
        x_msg.pose.position.y = 0
        x_msg.pose.position.z = x[1]
        x_msg.pose.orientation.w = 1
        x_pub.publish(x_msg)

        target_msg = Marker()
        target_msg.header.frame_id = 'world'
        target_msg.action = Marker.ADD
        target_msg.type = Marker.SPHERE
        target_msg.scale.x = s
        target_msg.scale.y = s
        target_msg.scale.z = s
        target_msg.color.a = 1.0
        target_msg.color.g = 1.0
        target_msg.pose.position.x = target[0]
        target_msg.pose.position.y = 0
        target_msg.pose.position.z = target[1]
        target_msg.pose.orientation.w = 1
        target_pub.publish(target_msg)

    iterative_projection(initial_value=tf.convert_to_tensor([0, 0], tf.float32),
                         target=tf.convert_to_tensor([1, 1], tf.float32),
                         n=100,
                         m=50,
                         m_last=100,
                         step_towards_target=step_x_towards_target,
                         project_opt=ProjectOpt(),
                         viz_func=viz_func)


if __name__ == '__main__':
    main()
