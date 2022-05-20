import resnet_model
import mem_util
import memory_saving_gradients
import tensorflow.contrib.graph_editor as ge
import tensorflow as tf
import pytest
import numpy as np
import math
"""Benchmark for memory reduction in deep resnet."""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description='deep resnet benchmark')
parser.add_argument('--name', type=str, default='deep',
                     help="name of benchmark run")
parser.add_argument('--min_blocks', type=int, default=50,
                     help="maximum number of blocks to add to resnet")
parser.add_argument('--max_blocks', type=int, default=102,
                     help="maximum number of blocks to add to resnet")
parser.add_argument('--outdir', type=str, default='.',
                     help="where to save results")
parser.add_argument('--disable_batch_norm', type=int, default=0,
                     help="where to save results")
args = parser.parse_args()

module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_path+'/..')

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # autotune adds random memory spikes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silence tf init messages


pytestmark = pytest.mark.skipif(not tf.test.is_gpu_available(),
                                reason="needs gpu")
resnet_model._DISABLE_BATCH_NORM = bool(args.disable_batch_norm)

# add_2:0, add_7:0, add_12:0, add_17:0, add_22:0, add_27:0, add_32:0, add_37:0, add_42:0, add_47:0, add_52:0, add_57:0,
USE_TINY = False

# resnet parameters
HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
if USE_TINY:
  BATCH_SIZE = 10
else:
  BATCH_SIZE = 4096
_WEIGHT_DECAY = 2e-4
_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9


# debug parameters
DUMP_GRAPHDEF = False


def profile(run_metadata, epoch=0):
    with open('profs/timeline_step' + str(epoch) + '.json', 'w') as f:
        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        f.write(chrome_trace)


def graph_to_dot(graph):
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot


def create_session():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(
      optimizer_options=optimizer_options))
  #  config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  config.graph_options.place_pruned_graph = True

  # fareed
  dot_rep = graph_to_dot(tf.get_default_graph())
  with open('profs/wrn.dot', 'w') as fwr:
      fwr.write(str(dot_rep))
  # trace_level=tf.RunOptions.FULL_TRACE,
  options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
  run_metadata = tf.RunMetadata()

  operations_tensors = {}
  operations_mem_forwarding = []
  operations_attributes = {}
  operations_names = tf.get_default_graph().get_operations()
  count1 = 0
  count2 = 0

  for operation in operations_names:
      # --------------------------
      operation_name = operation.name
      str_ = ''
      for input_ in operation.inputs:
          # print(input_.name)
          str_ += input_.name + '::'
      # print(str_)
      if len(str_) > 0:
          str_ = operation_name + '::' + str_
          operations_mem_forwarding.append(str_)
      # ----------------------------
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
  print(count1)
  print(count2)

  with open('./profs/op_mem_for.txt', 'w') as f:
      for forwarding in operations_mem_forwarding:
          f.write(forwarding + '\n')

  with open('./profs/tensors_sz_32.txt', 'w') as f:
      for tensor, size in operations_tensors.items():
          f.write('"' + tensor + '"::' + str(size) + '\n')

  with open('./profs/operations_attributes.txt', 'w') as f:
      for op, attrs in operations_attributes.items():
          strr = op
          for attr in attrs:
              strr += '::' + str(attr)
          strr += '\n'
          f.write(strr)

  return tf.Session(config=config)


def create_loss():
  """Creates loss tensor for resnet model."""
  images = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, DEPTH))
  labels = tf.random_uniform((BATCH_SIZE, NUM_CLASSES))
  # channels_last for CPU
  if USE_TINY:
    network = resnet_model.tiny_cifar10_resnet_v2_generator(
        RESNET_SIZE, NUM_CLASSES, data_format='channels_last')
  else:
    network = resnet_model.cifar10_resnet_v2_generator(
        RESNET_SIZE, NUM_CLASSES, data_format='channels_last')
  inputs = tf.reshape(images, [BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
  logits = network(inputs, True)
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                  onehot_labels=labels)
  l2_penalty = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  loss = cross_entropy + _WEIGHT_DECAY * l2_penalty
  return loss


GLOBAL_PROFILE = True
DUMP_TIMELINES = False
run_metadata = True


def sessrun(*args, **kwargs):
  global sess, run_metadata

  if not GLOBAL_PROFILE:
    return sess.run(*args, **kwargs)

  run_metadata = tf.RunMetadata()

  kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)

  if frd == 1:
    profile(run_metadata, step)

    if step == 13:
        options_mem = tf.profiler.ProfileOptionBuilder.time_and_memory()
        options_mem["min_bytes"] = 0
        options_mem["min_micros"] = 0
        options_mem["output"] = 'file:outfile=./profs/mem.txt'
        options_mem["select"] = ("bytes", "peak_bytes", "output_bytes",
                                 "residual_bytes")
        mem = tf.profiler.profile(tf.get_default_graph(
        ), run_meta=run_metadata, cmd="scope", options=options_mem)
        with open('profs/mem_2.txt', 'w') as f:
            f.write(str(mem))

  first_entry=args[0]
  if isinstance(first_entry, list):
    if len(first_entry) == 0 and len(args) == 1:
      return None
    first_entry=first_entry[0]

  if DUMP_TIMELINES:
    name=first_entry.name
    name=name.replace('/', '-')

    tl=timeline.Timeline(run_metadata.step_stats)
    ctf=tl.generate_chrome_trace_format()
    with open('timelines/%s.json' % (name,), 'w') as f:
      f.write(ctf)
    with open('timelines/%s.pbtxt' % (name,), 'w') as f:
      f.write(str(run_metadata))

  return result
frd=0
RESNET_SIZE=-1
def memory_test(resnet_blocks):
  """Evaluates gradient, returns memory in MB's and gradient eval time in
  seconds."""
  global sess, RESNET_SIZE

  RESNET_SIZE=resnet_blocks*6+2

  start_time0=time.perf_counter()
  tf.reset_default_graph()

  loss=create_loss()

  start_time=time.perf_counter()
  grads=tf.group(tf.gradients(loss, tf.trainable_variables()))

  sess=create_session()
  sessrun(tf.global_variables_initializer())
  times=[]
  memories=[]
  global frd
  for i in range(3):
    start_time=time.perf_counter()
    t0=time.time()
    frd=i
    sessrun(grads)
    print(time.time() - t0)
    elapsed_time=time.perf_counter() - start_time
    times.append(elapsed_time)
    mem_use=mem_util.peak_memory(run_metadata)['/gpu:0']/1e6
    memories.append(mem_use)
  exit()
  return np.min(memories), np.min(times)

class BufferedWriter:
  """Class that aggregates multiple writes and flushes periodically."""

  def __init__(self, outfn, save_every_secs=10):
    self.outfn=outfn
    self.last_save_ts=time.perf_counter()
    self.write_buffer=[]
    self.save_every_secs=save_every_secs

  def write(self, line):
    self.write_buffer.append(line)
    if time.perf_counter() - self.last_save_ts > self.save_every_secs:
      self.last_save_ts=time.perf_counter()
      with open(self.outfn, "a") as myfile:
        for line in self.write_buffer:
          myfile.write(line)
      self.write_buffer=[]

  def flush(self):
    with open(self.outfn, "a") as myfile:
      for line in self.write_buffer:
        myfile.write(line)
    self.write_buffer=[]

  def __del__(self):
    self.flush()


def main():
  old_gradients=tf.gradients

  # automatic checkpoint selection
  def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys,
                                             checkpoints='memory', **kwargs)
  print("Running with checkpoints")
  tf.__dict__["gradients"]=gradients_memory
  memories, times=[], []

  memory_f=BufferedWriter(args.outdir+'/'+args.name+'-opt-memory.csv')
  time_f=BufferedWriter(args.outdir+'/'+args.name+'-opt-time.csv')
  time2_f=BufferedWriter(args.outdir+'/'+args.name+'-opt-time2.csv')

  outf=open(args.outdir+'/'+args.name+'.csv', 'w')
  for i in range(args.min_blocks, args.max_blocks):
#    try:
      time0=time.time()
      memory_cost, time_cost=memory_test(i)
      time2=time.time()-time0
      print("%-10d %10d  %.2f seconds" % (i, memory_cost, time2))
      memory_f.write(str(memory_cost)+'\n')
      time_f.write(str(time_cost)+'\n')
      time2_f.write(str(time2)+'\n')

#    except Exception as e:
#      print("failed")
#      break

#    memories.append(memory_cost)
#    times.append(time_cost)


    #  def tostr(l): return [str(e) for e in l]
    #  outf.write(','.join(str(i) for i in range(1, args.max_blocks))+'\n')
    #  outf.write(','.join(tostr(memories))+'\n')
    #  outf.write(','.join(tostr(times))+'\n')


  memory_f=BufferedWriter(args.outdir+'/'+args.name+'-reg-memory.csv')
  time_f=BufferedWriter(args.outdir+'/'+args.name+'-reg-time.csv')
  time2_f=BufferedWriter(args.outdir+'/'+args.name+'-reg-time2.csv')

  # restore old gradients
  print("Running without checkpoints")
  tf.__dict__["gradients"]=old_gradients
  memories, times=[], []
  for i in range(args.min_blocks, args.max_blocks):
    try:
      time0=time.time()
      memory_cost, time_cost=memory_test(i)
      time2=time.time()-time0
      print("%-10d %10d  %5.2f seconds" % (i, memory_cost, time.time()-time0))
      memory_f.write(str(memory_cost)+'\n')
      time_f.write(str(time_cost)+'\n')
      time2_f.write(str(time2)+'\n')
    except Exception as e:
      print("failed")
      break
    memories.append(memory_cost)
    times.append(time_cost)


if __name__ == '__main__':
  main()
