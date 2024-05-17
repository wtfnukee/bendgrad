def Random seed
  seed: Int
  new(seed): Random { seed: seed }
  next(self):
    a = 1664525
    c = 1013904223
    m = 2^32
    new_seed = (a * self.seed + c) % m
    Random { seed: new_seed }
  uniform(self, low, high):
    self = self.next()
    value = low + (high - low) * (self.seed / 2^32)
    (value, self)
