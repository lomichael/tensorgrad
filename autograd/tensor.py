class Tensor:
    def __init__(self, data, creators=None, creation_op=None):
        self.data = data
        self.grad = None
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}

        if creators is not None:
            for c in creators:
                if self not in c.children:
                    c.children[self] = 1
                else:
                    c.children[self] += 1

    def backward(self, grad=None, grad_origin=None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if grad_origin is not None:
            if self.children[grad_origin] == 0:
                raise ValueError("Cannot backprop more than once")
            else:
                self.children[grad_origin] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            selfgrad += grad

        if self.creators is not None and (grad_origin is None or all(value == 0 for value in self.children.values())):
            if self.creation_op == "add":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)
            elif self.creation_op == "mul":
                new_grad = self.grad * self.creators[1]
                self.creators[0].backward(new_grad, self)
                new_grad = self.grad * self.creators[0]
                self.creators[1].backward(new_grad, self)

        def __add__(self, other):
            if not isinstance(other, Tensor):
                other = Tensor(other)
            return Tensor(self.data + other.data, creators=[self, other], creation_op="add")

        def __mul__(self, other):
            if not isintance(other, Tensor):
                other = Tensor(other)
            return Tensor(self.data * other.data, creators=[self, other], creation_op="mul")

        def __repr__(self):
            return f"Tensor of shape {self.data.shape} with data: {self.data}"
