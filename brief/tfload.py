import tensorflow as tf



def load1():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

    result = v1 + v2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './saverfiles/tfsave.ckpt')
        print(sess.run(result))
        

def load2():
    saver = tf.train.import_meta_graph('./saverfiles/tfsave.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess, './saverfiles/tfsave.ckpt')
        print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
       
def main(argv=None):
    load2()

# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()