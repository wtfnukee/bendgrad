def Value data grad _children _op _backward _prev
  data: Number
  grad: Number
  _children: List(Value)
  _op: String
  _backward: () -> ()
  _prev: Set(Value)
  new(data, _children = [], _op = ""): 
    Value {
      data: data,
      grad: 0,
      _children: _children,
      _op: _op,
      _backward: fn() => {},
      _prev: set(_children)
    }
  add(self, other):
    other_val = match other {
      case _: Value -> other
      default -> Value.new(other)
    }
    out_val = Value.new(self.data + other_val.data, [self, other_val], "+")
    out_val._backward = fn():
      self.grad = self.grad + out_val.grad
      other_val.grad = other_val.grad + out_val.grad
    out_val
  mul(self, other):
    other_val = match other {
      case _: Value -> other
      default -> Value.new(other)
    }
    out_val = Value.new(self.data * other_val.data, [self, other_val], "*")
    out_val._backward = fn():
      self.grad = self.grad + other_val.data * out_val.grad
      other_val.grad = other_val.grad + self.data * out_val.grad
    out_val
  pow(self, other):
    assert other.isInstanceOf(Int) or other.isInstanceOf(Float), "only supporting int/float powers for now"
    out_val = Value.new(self.data ** other, [self], "**" + str(other))
    out_val._backward = fn():
      self.grad = self.grad + (other * (self.data ** (other - 1))) * out_val.grad
    out_val
  relu(self):
    out_val = Value.new(if self.data < 0 then 0 else self.data, [self], "ReLU")
    out_val._backward = fn():
      self.grad = self.grad + (if out_val.data > 0 then out_val.grad else 0)
    out_val
  backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if not visited.contains(v):
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reverse(topo):
      v._backward()
  neg(self):
    self * -1
  radd(self, other):
    self + other
  sub(self, other):
    self + (-other)
  rsub(self, other):
    other + (-self)
  rmul(self, other):
    self * other
  truediv(self, other):
    self * other ** -1
  rtruediv(self, other):
    other * self ** -1
  repr(self):
    "Value(data=" + str(self.data) + ", grad=" + str(self.grad) + ")"
