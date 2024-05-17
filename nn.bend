import "random.bend" exposing (Randoms)
import "engine.bend" exposing (Value)

def Module
  zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  parameters(self):
    []

def Neuron w b nonlin
  w: List(Value)
  b: Value
  nonlin: Bool
  new(nin, nonlin=True, seed):
    rand = Random.new(seed)
    ws = []
    for i in range(0, nin):
      (val, rand) = rand.uniform(-1, 1)
      ws.append(Value.new(val))
    Neuron {
      w: ws,
      b: Value.new(0),
      nonlin: nonlin
    }
  call(self, x):
    act = foldl (fn (wi, xi, acc) => wi * xi + acc) self.b (zip(self.w, x))
    if self.nonlin then act.relu() else act
  parameters(self):
    self.w ++ [self.b]
  repr(self):
    (if self.nonlin then "ReLU" else "Linear") ++ "Neuron(" ++ str(len(self.w)) ++ ")"

def Layer neurons
  neurons: List(Neuron)
  new(nin, nout, **kwargs):
    ns = []
    seed = 42
    for i in range(0, nout):
      ns.append(Neuron.new(nin, kwargs.nonlin, seed))
      seed = seed + 1
    Layer { neurons: ns }
  call(self, x):
    out = [n.call(x) for n in self.neurons]
    if len(out) == 1 then out[0] else out
  parameters(self):
    flatten([n.parameters() for n in self.neurons])
  repr(self):
    "Layer of [" ++ join(", ", [str(n) for n in self.neurons]) ++ "]"

def MLP layers
  layers: List(Layer)
  new(nin, nouts):
    sz = [nin] ++ nouts
    ls = []
    for i in range(0, len(nouts)):
      ls.append(Layer.new(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)))
    MLP { layers: ls }
  call(self, x):
    for layer in self.layers:
      x = layer.call(x)
    x
  parameters(self):
    flatten([layer.parameters() for layer in self.layers])
  repr(self):
    "MLP of [" ++ join(", ", [str(layer) for layer in self.layers]) ++ "]"
