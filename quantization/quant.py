import math
import copy
import torch
from torch import nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
from torch.autograd.function import InplaceFunction, Function
import torch.nn.functional as F
#from IPython import embed

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    v = v.item()
    #print("max v:{}".format(v))
    if(v<1):
        si = 0
    else:
        si = math.ceil(math.log(v+1e-12,2))
        threshold = 2**(si-1)
        ratio = torch.sum(abs_value >= threshold).item() / abs_value.size()[0]
        if ratio < 0.01:
            si = si-1
    return si

def linear_quantize(input, sf, bits):
    if sf<0:
        sf = 0
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def compute_delta(input, bits, overflow_rate=0):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    v = v.item()
    si = math.ceil(math.log(v+1e-12,2))
    sf = bits - 1 - si
    delta = math.pow(2.0, -sf)
    return delta

def linear_quantize2(input, bits, delta):
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    input = input / delta + 0.5
    #rounded = torch.floor(input)
    rounded = QuantOp_floor.apply(input)
    #clipped_value = torch.clamp(rounded, min_val, max_val)
    clipped_value = QuantOp_clamp.apply(rounded, min_val, max_val)
    return clipped_value


class QuantOp_floor(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.floor(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantOp_clamp(Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        output = torch.clamp(input, min_val, max_val)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()
    #print min_val,max_val
    max_val = float(max_val.data.cpu().numpy())
    min_val = float(min_val.data.cpu().numpy())

    input_rescale = (input - min_val) / (max_val - min_val)
    #print "input_scale\n",input_rescale
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) # quantized int value
    #print "v\n",v
    v =  v / n * (max_val - min_val) + min_val # dequantized float value
    return v

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input_abs = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input_abs, bits)
    v = torch.exp(v) * s
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input_abs = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input_abs, sf, bits)
    v = torch.exp(v) * s
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    return v

def scale_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()
    max_val = float(max_val.data.cpu().numpy())
    min_val = float(min_val.data.cpu().numpy())
    #print('min_val:\n{},\nmax_val:\n{}'.format(min_val,max_val))
    n = math.pow(2.0, bits) - 1
    S = (max_val - min_val) / n
    Z = round(n*min_val / (min_val - max_val))
    input_rescale = (input - min_val) / (max_val - min_val)
    v = torch.round(input_rescale * n) # quantized int value
    return v, S, Z

def bn2conv(model): # map bn weight to conv 
    r""" conv layer must be arranged before bn layer !!!"""
    if isinstance(model,nn.Sequential):
        ikv = enumerate(model._modules.items())
        for i,(k,v) in ikv:
            if isinstance(v,nn.Conv2d):
                key,bn = next(ikv)[1]
                if isinstance(bn, nn.BatchNorm2d):
                    if bn.affine:
                        a = bn.weight / torch.sqrt(bn.running_var+bn.eps)
                        b = - bn.weight * bn.running_mean / torch.sqrt(bn.running_var+bn.eps) + bn.bias
                    else:
                        a = 1.0 / torch.sqrt(bn.running_var+bn.eps)
                        b = - bn.running_mean / torch.sqrt(bn.running_var+bn.eps)
                    v.weight = Parameter( v.weight * a.reshape(v.out_channels,1,1,1) )
                    v.bias   = Parameter(b)
                    model._modules[key] = nn.Sequential()
            else:
                bn2conv(v)
    else:
        for k,v in model._modules.items():
            bn2conv(v)

def replace_bn(model): # replace bn with depth-wise Conv2d
    if isinstance(model, nn.Sequential):
        for k,v in model._modules.items():
            if(isinstance(v, nn.BatchNorm2d)):
                C = v.num_features
                bn_new = nn.Conv2d(C, C, 1, groups=C)
                if v.affine:
                    a = v.weight / torch.sqrt(v.running_var+v.eps)
                    b = - v.weight * v.running_mean / torch.sqrt(v.running_var+v.eps) + v.bias
                else:
                    a = 1.0 / torch.sqrt(v.running_var+v.eps)
                    b = - v.running_mean / torch.sqrt(v.running_var+v.eps)
                bn_new.weight = Parameter(a.reshape(C,1,1,1))
                bn_new.bias   = Parameter(b)
                model._modules[k] = bn_new
            else:
                replace_bn(v)
    else:
        for k,v in model._modules.items():
            replace_bn(v)

def add_counter(model, counter):
    if isinstance(model, nn.Sequential):
        for k,v in model._modules.items():
            if("quant" in k):
                v._counter = counter
            else:
                add_counter(v, counter)
    else:
        for k,v in model._modules.items():
            add_counter(v, counter)

def combine_cb(model, param_bits, fwd_bits, counter): # combine conv and bn
    r""" conv layer must be arranged before bn layer !!!"""
    if isinstance(model,nn.Sequential):
        l = OrderedDict()
        ikv = enumerate(model._modules.items())
        for i,(k,v) in ikv:
            
            if isinstance(v,nn.Conv2d):
                key,bn = next(ikv)[1]
                if isinstance(bn, nn.BatchNorm2d):
                    quant_layer = Conv_BnQuant(name='{}_quant'.format(k), 
                                                param_bits=param_bits, 
                                                fwd_bits=fwd_bits,
                                                conv=v, 
                                                bn=bn,
                                                counter=counter)
                    l['{}_convbnquant'.format(k)] = quant_layer
            
            elif isinstance(v, nn.Linear):
                quant_layer = Linear_Quant(name='{}_quant'.format(k), 
                                                param_bits=param_bits, 
                                                fwd_bits=fwd_bits,
                                                module=v, 
                                                counter=counter)
                l['{}_linearquant'.format(k)] = quant_layer
            else:
                l[k] = combine_cb(v, param_bits, fwd_bits, counter)
        m = nn.Sequential(l)
        return m
    else:
        for k,v in model._modules.items():
            model._modules[k] = combine_cb(v, param_bits, fwd_bits, counter)
        return model


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter
    
    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            self.sf = max(self.sf, 0)
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class ScaleQuant(nn.Module):
    def __init__(self, name, bits, module, counter=10):
        super(ScaleQuant, self).__init__()
        self.name = name
        self._counter = counter
        self.max_cnt = counter
        self.bits = bits
        self.module = copy.deepcopy(module)
        self.m1 = copy.deepcopy(module)
        self.m2 = copy.deepcopy(module)
        self.m3 = copy.deepcopy(module)
        self.quant_module = copy.deepcopy(module)

        quant_weight, self.S1, self.Z1 = scale_quantize(self.module.weight, self.bits)
        #print("quant_weight:\n{}".format(quant_weight.view(-1).sort(descending=True)[0]))
        self.quant_module.weight = Parameter(quant_weight)
        self.bias = module.bias
        self.S2 = 0
        self.Z2 = 0
        self.S3 = 0
        self.Z3 = 0

        self.m1.weight = Parameter(quant_weight)
        self.m1.bias = None
        self.m2.weight = Parameter(self.Z1*torch.ones_like(module.weight))
        self.m2.bias = None

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            _, S2, Z2 = scale_quantize(input, self.bits)
            #self.S2, self.Z2  = S2, Z2
            self.S2 = (S2+self.S2*(self.max_cnt-self._counter))/(self.max_cnt-self._counter+1)
            self.Z2 = round((Z2+self.Z2*(self.max_cnt-self._counter))/(self.max_cnt-self._counter+1))
            #print("S2:{},Z2:{}".format(self.S2,self.Z2))
            output = self.module(input)
            _, S3, Z3 = scale_quantize(output, self.bits)
            self.S3, self.Z3  = S3, Z3
            #self.S3 = (S3+self.S3*(self.max_cnt-self._counter))/(self.max_cnt-self._counter+1)
            #self.Z3 = round((Z3+self.Z3*(self.max_cnt-self._counter))/(self.max_cnt-self._counter+1))
            return output
        else:
            self.m3.weight = Parameter(self.Z1*self.Z2*torch.ones_like(self.module.weight))
            self.m3.bias = None
            self.quant_module.bias = Parameter( torch.round(self.bias/(self.S1*self.S2)) )
            output_unquant = self.module(input)

            #print("S1:{},Z2:{}".format(self.S2,self.Z2))
            #print('unqaunt input:')
            #print(input.view(-1).sort(dim=0,descending=True)[0])
            
            input = torch.round(input/self.S2+self.Z2)
            input = torch.clamp(input,0,2**self.bits-1)

            #print('quant input:')
            #print(input.view(-1).sort(dim=0,descending=True)[0])
            
            q1q2 = self.quant_module(input)
            #print(q1q2.view(-1).sort(descending=True)[0])
            q1Z2 = self.m1( self.Z2*torch.ones_like(input) )
            #print(q1Z2.view(-1).sort(descending=True)[0])
            q2Z1 = self.m2(input)
            #print(q2Z1.view(-1).sort(descending=True)[0])
            Z1Z2 = self.m3( torch.ones_like(input) )
            #print(Z1Z2.view(-1).sort(descending=True)[0])
            output = q1q2 - (q1Z2 + q2Z1) + Z1Z2
            output = self.S1 * self.S2 * output
            #print('unquant output:')
            #print(output_unquant.view(-1).sort(dim=0,descending=True)[0])
            #print('quant output:')
            #print(output.view(-1).sort(dim=0,descending=True)[0])
            #print('\n')
            return output

class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)

class LinearQuant2(nn.Module):
    def __init__(self, name, param_bits, fwd_bits, module, delta_b=None, overflow_rate=0.0, counter=10):
        super(LinearQuant2, self).__init__()
        self.delta_b = delta_b
        self.delta_list = []
        self.name = name
        self._counter = counter
        self.param_bits = param_bits
        self.fwd_bits = fwd_bits
        self.overflow_rate = overflow_rate
        self.module = copy.deepcopy(module) 
        self.q_module = copy.deepcopy(module) #copy module shape

    @property
    def counter(self):
        return self._counter
    
    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            #print(self._counter)
            delta_b = compute_delta(input, self.fwd_bits, self.overflow_rate)
            self.delta_list.append(delta_b)
            output = self.module(input)
            if self._counter == 0:
                self.delta_list.sort()
                half = len(self.delta_list) // 2
                self.delta_b = self.delta_list[half]
            return output
        else:
            #self.module(input)
            self.delta_a = compute_delta(self.module.weight, self.param_bits, self.overflow_rate)
            quant_weight = linear_quantize2(self.module.weight, self.param_bits, self.delta_a)
            self.q_module.weight.data = quant_weight

            quant_bias =  torch.round( self.module.bias / (self.delta_a*self.delta_b) )
            self.q_module.bias.data = quant_bias

            q_input = linear_quantize2(input, self.fwd_bits, self.delta_b)
            q_output = self.q_module(q_input)
            q_output = q_output * self.delta_a * self.delta_b
            return q_output


class QuantOp_Linear(Function):
    @staticmethod
    def forward(ctx, input, q_input):
        return q_input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Linear_Quant(nn.Module):
    def __init__(self, name, param_bits, fwd_bits, module, delta_b=None, overflow_rate=0.0, counter=10):
        super(Linear_Quant, self).__init__()
        self.delta_b = delta_b
        self.delta_a = None
        self.delta_list = []
        self.name = name
        self._counter = counter
        self.param_bits = param_bits
        self.fwd_bits = fwd_bits
        self.overflow_rate = overflow_rate
        #self.module = copy.deepcopy(module)
        self.weight = module.weight
        self.bias = module.bias
        #self.q_module = copy.deepcopy(module) #copy module shape

    @property
    def counter(self):
        return self._counter
    
    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            delta_b = compute_delta(input, self.fwd_bits, self.overflow_rate)
            self.delta_list.append(delta_b)
            if self._counter == 0:
                self.delta_list.sort()
                half = len(self.delta_list) // 2
                self.delta_b = self.delta_list[half]
            output = F.linear(input, self.weight, self.bias)
            #output = self.module(input)
            return output
        else:
            self.delta_a = compute_delta(self.weight, self.param_bits, self.overflow_rate)
            q_weight = self.delta_a * linear_quantize2(self.weight, self.param_bits, self.delta_a)

            q_bias =  QuantOp_floor.apply( self.bias / (self.delta_a*self.delta_b) ) * self.delta_a * self.delta_b
            

            q_input = self.delta_b * linear_quantize2(input, self.fwd_bits, self.delta_b)
            q_output = F.linear(q_input, q_weight, q_bias)
            #print(q_weight.view(-1).topk(5))
            #q_output = q_output * self.delta_a * self.delta_b
            return q_output

class Conv_BnQuant(nn.Module):
    def __init__(self, name, param_bits, fwd_bits, conv, bn, delta_b=None, overflow_rate=0.0, counter=10):
        super(Conv_BnQuant, self).__init__()
        self.delta_b = delta_b
        self.delta_list = []
        self.name = name
        self._counter = counter
        self.param_bits = param_bits
        self.fwd_bits = fwd_bits
        self.overflow_rate = overflow_rate
        #self.conv = copy.deepcopy(conv)
        self.bn = copy.deepcopy(bn)

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.conv_out_channels = conv.out_channels

        self.conv_weight = conv.weight
        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.bn_affine = bn.affine
        self.eps = bn.eps


    @property
    def counter(self):
        return self._counter
    
    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            print("counter:{}".format(self._counter))
            delta_b = compute_delta(input, self.fwd_bits, self.overflow_rate)
            self.delta_list.append(delta_b)
            if self._counter == 0:
                self.delta_list.sort()
                half = len(self.delta_list) // 2
                self.delta_b = self.delta_list[half]
            self.bn.weight = Parameter(self.bn_weight)
            self.bn.bias = Parameter(self.bn_bias)
            output = F.conv2d(input, self.conv_weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
            output = self.bn(output)
            return output
        else:
            #self.bn(self.conv(input))
            y_t = F.conv2d(input, self.conv_weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
            self.bn(y_t)
            if self.bn_affine:
                a = self.bn_weight / torch.sqrt(self.bn.running_var + self.eps)
                b = - self.bn_weight * self.bn.running_mean / torch.sqrt(self.bn.running_var + self.eps) + self.bn_bias
            else:
                a = 1.0 / torch.sqrt(self.bn.running_var + self.eps)
                b = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.eps)
            a_extension = a.reshape(self.conv_out_channels,1,1,1)
            combine_weight = self.conv_weight * a_extension   #combine conv param with bn param to conv
            #print(combine_weight.view(-1).topk(5))
            delta_a      = compute_delta(combine_weight, self.param_bits, self.overflow_rate)
            quant_weight = delta_a * linear_quantize2(combine_weight, self.param_bits, delta_a)
            quant_bias = QuantOp_floor.apply(b / (delta_a*self.delta_b)) * delta_a * self.delta_b

            q_input = self.delta_b * linear_quantize2(input, self.fwd_bits, self.delta_b)
            q_output = F.conv2d(q_input, quant_weight, quant_bias, self.stride,
                        self.padding, self.dilation, self.groups)
            #q_output = q_output * delta_a * self.delta_b
            #print(quant_weight.view(-1).topk(5))
            
            return q_output

def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original model has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.AvgPool2d)):
                l[k] = v
                if   type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer
            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        return model

def duplicate_model_with_scalequant(model, bits, counter=10):
    """assume that original model has at least a nn.Sequential"""
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.AvgPool2d)):
                quant_layer = ScaleQuant(name='{}_quant'.format(k), bits=bits, 
                module=v, counter=counter)
                l['{}_scalequant'.format(k)] = quant_layer
            else:
                l[k] = duplicate_model_with_scalequant(v, bits, counter)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_scalequant(v, bits, counter)
        return model

def duplicate_model_with_linearquant(model, param_bits, fwd_bits, overflow_rate=0.0, counter=10):
    """assume that original model has at least a nn.Sequential"""
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                quant_layer = LinearQuant2(name='{}_quant'.format(k), 
                param_bits=param_bits, fwd_bits=fwd_bits,
                module=v, counter=counter)
                l['{}_linearquant2'.format(k)] = quant_layer
            else:
                l[k] = duplicate_model_with_linearquant(v, param_bits, fwd_bits, overflow_rate, counter)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_linearquant(v, param_bits, fwd_bits, overflow_rate, counter)
        return model

# return quantized model
def quantized_model(model, args):
    # replace bn layer
    if args.replace_bn:
        replace_bn(model)

    # map bn to conv
    if args.map_bn:
        bn2conv(model)

    print("=================quantize parameters==================")
    if args.param_bits < 32:
        state_dict = model.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k: # quantize bn layer
                #print("k:{}, v:\n{}".format(k,v))
                if args.bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = args.bn_bits
            else:
                bits = args.param_bits

            if args.quant_method == 'linear':
                sf = bits - 1. - compute_integral_part(v, overflow_rate=args.overflow_rate)
                # sf stands for float bits
                v_quant  = linear_quantize(v, sf, bits=bits)
                #if 'bias' in k:
                    #print("{}, sf:{}, quantized value:\n{}".format(k,sf, v_quant.sort(dim=0, descending=True)[0]))
            elif args.quant_method == 'log':
                v_quant = log_minmax_quantize(v, bits=bits)
            elif args.quant_method == 'minmax':
                v_quant = min_max_quantize(v, bits=bits)
            else:
                v_quant = tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
            # print("k={0:<35}, bits={1:<5}, sf={2:d>}".format(k,bits,sf))
        model.load_state_dict(state_dict_quant)
    print("======================================================")

    # quantize forward activation
    print("=================quantize activation==================")
    if args.fwd_bits < 32:
        model = duplicate_model_with_quant(model, 
                                            bits=args.fwd_bits, 
                                            overflow_rate=args.overflow_rate,
                                            counter=args.n_sample, 
                                            type=args.quant_method)
    return model