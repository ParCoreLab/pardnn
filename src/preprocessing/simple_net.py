from netrc import netrc
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
import time
from graphviz import Digraph 

from tensorflow.python.client import timeline
import tensorflow.compat.v1 as tf

from tensorflow.contrib import graph_editor as ge


#tf.compat.v1.disable_v2_behavior()

sz1 = 1024
sz2 = 4092
sz3 = 4096
#x = tf.Variable(tf.random_normal([sz1, sz2], seed=1234), name='x')
#y = tf.Variable(tf.random_normal([sz1, sz2], seed=1234), name='y')

"""with tf.name_scope("gpu_0"):
    with tf.device('/gpu:7'):
        z1 = tf.multiply(x, y)

with tf.name_scope("gpu_1"):
    with tf.device('/gpu:9'):
        z2 = tf.multiply(x, y)
        z2 = tf.multiply(z2, z2)"""  

def graph_to_dot(graph):
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label= n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot

with tf.device('/gpu:7'):
    z1 = tf.Variable(tf.random_normal([sz1, sz2], seed=1234), name='z1')
    z2 = tf.Variable(tf.random_normal([sz2, sz2], seed=1234), name='z2')
    z3 = tf.Variable(tf.random_normal([sz2, sz3], seed=1234), name='z3')
    cc = tf.Variable(0, name="cc", dtype = tf.int32)

""" with tf.device('/gpu:7'):
    z2_1, z2_2 = tf.split(z2, num_or_size_splits=2) 
    res1 = tf.matmul(z2_1, z3)
with tf.device('/gpu:9'):
    res2 = tf.matmul(z2_2, z3)

with tf.device('/gpu:7'):
    result3=tf.concat([res1, res2], 0)
    result = tf.reduce_sum(result3) """

with tf.device('/gpu:7'):
    result3 = tf.matmul(z2, z3, name="res3")
    result = tf.reduce_sum(result3, name="res")

""" with tf.device('/gpu:7'):
    z2_0, z2_1 = tf.split(tf.get_default_graph().get_operation_by_name("res3").inputs[0], num_or_size_splits=2, name="spl")
    res1 = tf.matmul(z2_0, tf.get_default_graph().get_operation_by_name("res3").inputs[1])
with tf.device('/gpu:9'):
    res2 = tf.matmul(z2_1, tf.get_default_graph().get_operation_by_name("res3").inputs[1])

with tf.device('/gpu:7'):
    res_t = tf.concat([res1, res2], 0)
    ge.detach_inputs(tf.get_default_graph().get_operation_by_name("res3"))
    ge.connect( ge.sgv(res_t), ge.sgv(tf.get_default_graph().get_operation_by_name("res")).remap_inputs([0]), True) """


#verification
""" with tf.device('/gpu:7'):
    result2=tf.matmul(z2, z3)
    result3 = tf.math.subtract(result2, result1)
    result = tf.reduce_sum(result3)
    result = tf.add(result, 1)
    tf.get_static_value(result, partial=False) """


init_op = tf.global_variables_initializer()


def profile(run_metadata, epoch=0):
    with open('./profs/timeline_step' + str(epoch) + '.json', 'w') as f:
        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        f.write(chrome_trace)
        
print(tf.__version__)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = \
tf.GPUOptions(per_process_gpu_memory_fraction=0.96) )) as sess:
    options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    run_metadata = tf.RunMetadata()
    
    dot_rep = graph_to_dot(tf.get_default_graph())
    with open('./A_dot.dot', 'w') as fwr:
        fwr.write(str(dot_rep))

    sess.run(init_op, run_metadata=run_metadata, options=options)
	
    for i in range(1, 13):
        time.sleep(1)
        if i == 12:
            options_mem = tf.profiler.ProfileOptionBuilder.time_and_memory()
            options_mem["min_bytes"] = 0
            options_mem["min_micros"] = 0
            options_mem["output"] = 'file:outfile=./mem.txt'
            options_mem["select"] = ("bytes", "peak_bytes", "output_bytes",
                                 "residual_bytes")
            values = sess.run(result, run_metadata=run_metadata, options=options)
            profile(run_metadata, 0)
            
            
            mem = tf.profiler.profile(tf.get_default_graph(), run_meta=run_metadata, cmd="scope", options=options_mem)
            with open('./mem2.txt', 'w') as f:
                f.write(str(mem))
        else:
            t0 = time.time()  
            values = sess.run(result, options = options)
            print(time.time() - t0)
        #print('fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
        



    operations_tensors = {}
    operations_attributes = {}
    operations_mem_forwarding = []
    operations_names = tf.get_default_graph().get_operations()
    count1 = 0
    count2 = 0

    for operation in operations_names:
        operation_name = operation.name
        operations_info = tf.get_default_graph(
        ).get_operation_by_name(operation_name).values()
        
        try:
            operations_attributes[operation_name] = []
            operations_attributes[operation_name].append(operation.type)
            operations_attributes[operation_name].append(tf.get_default_graph(
            ).get_tensor_by_name(operation_name + ':0').dtype._is_ref_dtype)
        except:
            pass
        if len(operations_info) > 0:
            if not (operations_info[0].shape.ndims is None):
                str_ = ''
                for input_ in operation.inputs:
                    #print(input_.name)
                    str_ += input_.name + '::'
                #print(str_)
                if len(str_) > 0:
                    str_ = operation_name + '::' +str_                                
                    operations_mem_forwarding.append(str_)
                operation_shape = operations_info[0].shape.as_list()
                operation_dtype_size = operations_info[0].dtype.size
                if not (operation_dtype_size is None):
                    operation_no_of_elements = 1
                    for dim in operation_shape:
                        if not(dim is None):
                            operation_no_of_elements = operation_no_of_elements * dim
                    total_size = operation_no_of_elements * operation_dtype_size
                    operations_tensors[operation_name] = total_size
                else:
                    count1 = count1 + 1
            else:
                count1 = count1 + 1
                operations_tensors[operation_name] = -1

            #   print('no shape_1: ' + operation_name)
            #  print('no shape_2: ' + str(operations_info))
            #  operation_namee = operation_name + ':0'
            # tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
            # print('no shape_3:' + str(tf.shape(tensor)))
            # print('no shape:' + str(tensor.get_shape()))

        else:
            # print('no info :' + operation_name)
            # operation_namee = operation.name + ':0'
            count2 = count2 + 1
            operations_tensors[operation_name] = -1

            # try:
            #   tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
            # print(tensor)
            # print(tf.shape(tensor))
            # except:
            # print('no tensor: ' + operation_namee)
    #print(count1)
    #print(count2)
    with open('./tensors_sz_32.txt', 'w') as f:
        for tensor, size in operations_tensors.items():
            f.write('"' + tensor + '"::' + str(size) + '\n')
    
    with open('./op_mem_for.txt', 'w') as f:
        for forwarding in operations_mem_forwarding:
            f.write(forwarding + '\n')
                        
    with open('./operations_attributes.txt', 'w') as f:
        for op, attrs in operations_attributes.items():
            strr = op
            for attr in attrs:
                strr += '::' + str(attr)
            strr += '\n'
            f.write(strr)
