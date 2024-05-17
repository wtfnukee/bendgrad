# bendgrad

Have you heard of [Bend](https://github.com/HigherOrderCO/Bend)? I didnt before yesterday evening. That is massively parallel, high-level programming language, that just a set of Python style bindings for Rust parallelization bindings.

Have you heard [micrograd](https://github.com/karpathy/micrograd)? That is, citing `A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API`. Sounds cool!

What sound even cooler? micrograd written in two day old language!

Im not sure I wrote it idiomatically and I cant really benchmark it, bc it doesnt even has IO, but Im looking forward to it.

Besides that, it doesnt has random generator, so I wrote some classic pseudorandom.

So, how do I use it? Other than installing Rust and Bend, clone this repo and add some neurons

```
import "nn.bend" exposing (MLP, Value)

def main():
  # Example usage
  mlp = MLP.new(3, [4, 4, 1])
  input_values = [Value.new(1.0), Value.new(2.0), Value.new(3.0)]
  output = mlp.call(input_values)
  print(output) # 🙏🙏🙏

main()
```