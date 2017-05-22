import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
class GraphUtilites(object):
    """ Utility for saving and reloading graph """
    def __init__(self):
        pass

    def freeze_graph(self,model_folder, frozen_file_name, out_node_names):
        """ A function to freeze model files 
        Args:
            model_folder (string) : full path of the folder where checkpoint
                                    files are stored
            frozen_file_name (string) : Name to be given to the frozen model

            out_node_names (comma separated string) : name of the output nodes to
                                                      saved in a graph
        Returns:
            (None) : Writes the protobuf file to the disk
        """
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print ("Freezing output graph name", input_checkpoint)
        print (input_checkpoint)
        out_graph = os.path.join(model_folder, frozen_file_name)
        # TODO: Make it genric to accept node as a function parameter
        output_node_name = "out_pred"
        clear_device = True
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                           clear_devices = True)
        
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        with tf.Session() as load_session:
            saver.restore(load_session, input_checkpoint)
            output_graph_def = graph_util.convert_variables_to_constants(load_session,
                                                      input_graph_def,
                                                      output_node_name.split(','))
            with tf.gfile.GFile(out_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print ("%d ops in the final graph." %len(output_graph_def.node))


    def load_graph(self,frozen_file_name):
        """ load the frozen model file
        Args:
            frozen_file_name (string) : relative path to the file to be loaded

        Returns:
            (tensorflow graph object) : returns the reconstructed graph object
        """
        with tf.gfile.GFile(frozen_file_name, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
        return graph
