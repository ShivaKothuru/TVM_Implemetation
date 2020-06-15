import onnx
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing


onnx_model = onnx.load('onnx models/monodepth2.onnx')

#import logging
#logging.basicConfig(level=logging.DEBUG) # to dump TVM IR after fusion

target = "cuda"
input_name = 'input'
shape_dict = {input_name: (1,3,192,640)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
print(mod)
with relay.build_config(opt_level=1):
    intrp = relay.build_module.create_executor('graph', mod, tvm.gpu(0), target)
graph, lib, params = relay.build(mod, target, params=params)


name = 'Monodepth'
graph_fn, mod_fn, params_fn = [name+ext for ext in ('.json','.tar','.params')]
lib.export_library(mod_fn)
with open(graph_fn, 'w') as f:
    f.write(graph)
with open(params_fn, 'wb') as f:
    f.write(relay.save_param_dict(params))
ctx = tvm.context(target, 0)



