# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn


class SgdParameterModifier:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.initial_values = [e.data for e in parameters]
        self.grads = None
        self.lr = lr

    def restore_initial_values(self):
        for parameter, initial_value in zip(self.parameters, self.initial_values):
            parameter.data = initial_value

    def set_grads(self):
        for parameter, grad in zip(self.parameters, self.grads):
            parameter.grad = grad

    def update_parameters(self, args):
        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(parameters=self.parameters,
                                     max_norm=args.clip_grad_norm,
                                     norm_type=float("inf"))

        is_none = self.grads is None
        if is_none:
            self.grads = []

        for i, p in enumerate(self.parameters):
            p.data.add_(-self.lr, p.grad)

            if is_none:
                self.grads.append(p.grad.clone())
            else:
                self.grads[i] += p.grad
            p.grad.zero_()
