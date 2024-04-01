#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import itertools
import platform

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.mil.types import builtin_to_string, nptype_from_builtin
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import random_gen, ssa_fn


class TestLinear:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([[-4.7182, 11.94], [-3.3939, 9.2166]], dtype=np.float32)
        weight_val = np.array([[1.2313, -0.095], [-1.4075, -0.8816]], dtype=np.float32)
        bias_val = np.array([1.0, 2.0], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [mb.linear(x=x, weight=weight_val, bias=bias_val)]

        expected_output_types = [(2, 2, types.fp32)]
        expected_outputs = [
            np.array([[-5.9438195, -1.8854373], [-4.054486, -1.3484411]], dtype=np.float32)
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = random_gen(shape=(2, 2), rand_min=-37, rand_max=64)
        weight_val = random_gen(shape=(2, 2), rand_min=-91, rand_max=84)
        bias_val = random_gen(shape=(2,), rand_min=0.0, rand_max=9.0)
        v = mb.linear(x=x_val, weight=weight_val, bias=bias_val)
        np.testing.assert_allclose(
            np.matmul(x_val, weight_val.T) + bias_val, v.val, atol=1e-04, rtol=1e-05
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(compute_units, backends, [2, 3, 5]),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, rank):
        if backend.backend == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.xfail("rdar://97398733 (TestLinear failing on mlprogram + GPU)")

        if (
            backend.backend == "neuralnetwork"
            and compute_unit != ct.ComputeUnit.CPU_ONLY
            and platform.machine() == "arm64"
            and rank == 5
        ):
            pytest.xfail(
                "rdar://98015195 ([M1 native tests] Some MIL unittests are failing on M1 native)"
            )

        x_shape = np.random.randint(low=1, high=3, size=(rank,))
        x_val = np.random.rand(*x_shape)
        out_channels = 3
        w_shape = np.array([out_channels, x_shape[-1]])
        weight_val = np.random.rand(*w_shape).astype(np.float32)
        bias_val = np.random.rand(out_channels).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
        }
        input_values = {"x": x_val}

        def build(x):
            return [mb.linear(x=x, weight=weight_val, bias=bias_val)]

        expected_outputs = [np.matmul(x_val, np.transpose(weight_val)) + bias_val]

        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs=expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, input_type",
        itertools.product(compute_units, backends, [types.int32, types.fp16, types.fp32]),
    )
    def test_default_bias_type(self, compute_unit, backend, input_type):
        # Test the default bias matches the dtype of x and weight.
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 2), dtype=types.fp32)],
            opset_version=backend.opset_version,
        )
        def prog(x):
            x = mb.cast(x=x, dtype=builtin_to_string(input_type))
            weight = np.random.rand(3, 2).astype(nptype_from_builtin(input_type))
            res = mb.linear(x=x, weight=weight)
            assert res.op.bias.val.dtype == nptype_from_builtin(input_type)
            return res


class TestMatMul:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([[-4.0, 13.0], [-3.0, 9.0]], dtype=np.float32)
        y_val = np.array([[1.0, -7.0], [-1.0, -8.0]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
            "y": mb.placeholder(shape=y_val.shape),
        }
        input_values = {"x": x_val, "y": y_val}

        def build(x, y):
            return [
                mb.matmul(x=x_val, y=y),
                mb.matmul(x=x, y=y_val),
                mb.matmul(x=x, y=y),
                mb.matmul(x=x, y=y, transpose_x=True, transpose_y=True),
                mb.matmul(x=x_val, y=y, transpose_x=True, transpose_y=True),
                mb.matmul(x=x, y=y_val, transpose_x=True, transpose_y=True),
                mb.matmul(x=x, y=y_val, transpose_x=True, transpose_y=False),
                mb.matmul(x=x, y=y_val, transpose_x=False, transpose_y=True),
            ]

        expected_output_types = [
            (2, 2, types.fp32),
            (2, 2, types.fp32),
            (2, 2, types.fp32),
            (2, 2, types.fp32),
            (2, 2, types.fp32),
            (2, 2, types.fp32),
            (2, 2, types.fp32),
            (2, 2, types.fp32),
        ]
        expected_outputs = [
            np.array([[-17.0, -76.0], [-12.0, -51.0]], dtype=np.float32),
            np.array([[-17.0, -76.0], [-12.0, -51.0]], dtype=np.float32),
            np.array([[-17.0, -76.0], [-12.0, -51.0]], dtype=np.float32),
            np.array([[17.0, 28.0], [-50.0, -85.0]], dtype=np.float32),
            np.array([[17.0, 28.0], [-50.0, -85.0]], dtype=np.float32),
            np.array([[17.0, 28.0], [-50.0, -85.0]], dtype=np.float32),
            np.array([[-1.0, 52.0], [4.0, -163.0]], dtype=np.float32),
            np.array([[-95.0, -100.0], [-66.0, -69.0]], dtype=np.float32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = random_gen(shape=(2, 2, 4), rand_min=-37, rand_max=64)
        y_val = random_gen(shape=(2, 4, 2), rand_min=-91, rand_max=84)
        v = mb.matmul(x=x_val, y=y_val)
        np.testing.assert_allclose(np.matmul(x_val, y_val), v.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [
                ((3, 2, 3, 4), (3, 2, 4, 5)),
                ((1, 1, 1, 3, 4), (1, 3, 2, 4, 5)),
                ((1, 3, 1, 2, 3), (1, 4, 3, 2)),
                ((1, 3, 4), (3, 2, 4, 6)),
                ((7, 4), (3, 9, 5, 4, 3)),
            ],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, shapes):
        shape_x, shape_y = shapes
        x_val = np.random.rand(*shape_x)
        y_val = np.random.rand(*shape_y)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
            "y": mb.placeholder(shape=y_val.shape),
        }
        input_values = {"x": x_val, "y": y_val}

        def build(x, y):
            return [mb.matmul(x=x, y=y, transpose_x=False, transpose_y=False)]

        expected_outputs = [np.matmul(x_val, y_val)]
        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape_x",
        itertools.product(
            compute_units,
            backends,
            [
                (5,),
                (2, 5),
                (2, 2, 5),
                (4, 3, 2, 5),
                (5, 4, 2, 3, 5),
            ],
        ),
    )
    def test_builder_y_rank_2_const(self, compute_unit, backend, shape_x):
        x_val = np.random.rand(*shape_x)
        y_val = np.random.rand(5, 10)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
        }
        input_values = {"x": x_val}

        def build(x):
            return [mb.matmul(x=x, y=y_val, transpose_x=False, transpose_y=False)]

        expected_outputs = [np.matmul(x_val, y_val)]
        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_transpose_y(self, compute_unit, backend):
        x_val = np.random.rand(3, 2, 7, 16)
        y_val = np.random.rand(3, 2, 5, 16)

        def build(x):
            return mb.matmul(x=x, y=y_val, transpose_x=False, transpose_y=True)

        expected_output = np.matmul(x_val, np.transpose(y_val, (0, 1, 3, 2)))
        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x_val.shape)},
            input_values={"x": x_val},
            expected_output_types=expected_output.shape + (types.fp32,),
            expected_outputs=expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestEinsum:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        equation = "abcd,adce->abce"

        x_val = np.arange(12).astype(np.float32).reshape((2, 1, 3, 2))
        y_val = np.arange(48).astype(np.float32).reshape((2, 2, 3, 4))
        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_val.shape),
            "y": mb.placeholder(shape=y_val.shape),
        }
        input_value_dict = {"x": x_val, "y": y_val}
        out_shape = list(x_val.shape)
        out_shape[-1] = y_val.shape[-1]
        expected_output_type = tuple(out_shape) + (types.fp32,)

        def build(x, y):
            return mb.einsum(values=(x, y), equation=equation)

        expected_output = np.einsum(equation, x_val, y_val)

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, rank, broadcast, backend",
        itertools.product(
            compute_units,
            [3, 4],
            [True, False],
            backends,
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, rank, broadcast, backend):
        equation = "abcd,adce->abce" if rank == 4 else "vnm,mno->vno"
        shape_x = np.random.randint(low=2, high=16, size=rank).astype(np.int32)
        shape_y = np.random.randint(low=2, high=12, size=rank).astype(np.int32)
        shape_y[-3] = shape_x[-1]
        shape_y[-2] = 1 if broadcast else shape_x[-2]
        if rank == 4:
            shape_x[-4] = 1 if broadcast else shape_y[-4]

        x_val = np.random.rand(*shape_x)
        y_val = np.random.rand(*shape_y)
        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_val.shape),
            "y": mb.placeholder(shape=y_val.shape),
        }

        input_value_dict = {"x": x_val, "y": y_val}
        out_shape = (
            [shape_y[-4], shape_x[-3], shape_x[-2], shape_y[-1]]
            if rank == 4
            else [shape_x[-3], shape_x[-2], shape_y[-1]]
        )
        expected_output_type = tuple(out_shape) + (types.fp32,)

        def build(x, y):
            return mb.einsum(values=(x, y), equation=equation)

        if rank == 3:
            expected_output = np.einsum(
                equation,
                np.broadcast_to(x_val, [shape_x[-3], shape_x[-2], shape_x[-1]]),
                np.broadcast_to(y_val, [shape_y[-3], shape_x[-2], shape_y[-1]]),
            )
        else:
            expected_output = np.einsum(
                equation,
                np.broadcast_to(x_val, [shape_y[-4], shape_x[-3], shape_x[-2], shape_x[-1]]),
                np.broadcast_to(y_val, [shape_y[-4], shape_y[-3], shape_x[-2], shape_y[-1]]),
            )

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.arange(6).astype(np.float32).reshape((1, 3, 2))
        y_val = np.arange(24).astype(np.float32).reshape((2, 3, 4))
        equation = "bcd,dce->bce"
        v = mb.einsum(values=(x_val, y_val), equation=equation)
        np.testing.assert_allclose(np.einsum(equation, x_val, y_val), v.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_symbolic_input_conv_and_einsum(self, backend):
        """
        Test a pattern of:

            %1 = conv_1(%x)
            %2 = conv_2(%x)
            %3 = transpose(%2, [0, 3, 2, 1])
            %4 = einsum(%1, %3)

        If ``%x`` has symbolic shape and ``conv_1, conv_2`` have the same
        configuration, the above program should pass the type inference.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 3, get_new_symbol(), get_new_symbol()), dtype=types.fp32)
            ],
            opset_version=backend.opset_version,
        )
        def prog(x):
            weight = np.random.rand(2, 3, 2, 2)
            conv_1 = mb.conv(x=x, weight=weight)
            conv_2 = mb.conv(x=x, weight=weight)
            conv_2_transpose = mb.transpose(x=conv_2, perm=[0, 3, 2, 1])
            return mb.einsum(values=(conv_1, conv_2_transpose), equation="abcd,adce->abce")

        assert prog is not None
