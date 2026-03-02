#pragma once
#ifndef PYTHON_GRAPH_HPP
#define PYTHON_GRAPH_HPP
#include "core/graph_builder.h"
#include "core/runtime.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infini {
void bind_graph_builder(py::module &m) {
    py::class_<GraphObj, std::shared_ptr<GraphObj>>(m, "Graph");
    // GraphBuilder
    py::class_<GraphBuilderObj>(m, "GraphBuilder")
        .def(py::init<Runtime>())
        .def("tensor", &GraphBuilderObj::tensor, py::arg("dims"),
             py::arg("dtype"), py::arg("stride") = py::none())
        .def("gemm", &GraphBuilderObj::gemm, py::arg("A"), py::arg("B"),
             py::arg("C"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("transA") = false, py::arg("transB") = false,
             py::arg("Y") = py::none())
        .def("clip", &GraphBuilderObj::clip, py::arg("X"), py::arg("min_val"),
             py::arg("max_val"), py::arg("Y") = py::none())
        .def("add", &GraphBuilderObj::add, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("sub", &GraphBuilderObj::sub, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("mul", &GraphBuilderObj::mul, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("conv", &GraphBuilderObj::conv, py::arg("x"), py::arg("w"),
             py::arg("b"), py::arg("pads"), py::arg("strides"),
             py::arg("dilations"), py::arg("Y") = py::none())
        .def("layernorm", &GraphBuilderObj::layernorm, py::arg("x"),
             py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f,
             py::arg("Y") = py::none())
        .def("lpnorm", &GraphBuilderObj::lpnorm, py::arg("x"), py::arg("axis"),
             py::arg("p") = 2, py::arg("eps") = 1e-12f,
             py::arg("Y") = py::none())
        .def("rmsnorm", &GraphBuilderObj::rmsnorm, py::arg("x"), py::arg("w"),
             py::arg("epsilon") = 1e-6f, py::arg("Y") = py::none())
        .def("softmax", &GraphBuilderObj::softmax, py::arg("x"),
             py::arg("axis") = -1, py::arg("Y") = py::none())
        .def("logsoftmax", &GraphBuilderObj::logsoftmax, py::arg("x"),
             py::arg("Y") = py::none())
        .def("relu", &GraphBuilderObj::relu, py::arg("x"),
             py::arg("Y") = py::none())
        .def("sigmoid", &GraphBuilderObj::sigmoid, py::arg("x"),
             py::arg("Y") = py::none())
        .def("silu", &GraphBuilderObj::silu, py::arg("x"),
             py::arg("Y") = py::none())
        .def("gelu", &GraphBuilderObj::gelu, py::arg("x"),
             py::arg("Y") = py::none())
        .def("softplus", &GraphBuilderObj::softplus, py::arg("x"),
             py::arg("Y") = py::none())
        .def("tanh", &GraphBuilderObj::tanh, py::arg("x"),
             py::arg("Y") = py::none())
        .def("to_string", &GraphBuilderObj::printGraph)
        .def_property_readonly("graph", &GraphBuilderObj::getGraph);
}

} // namespace infini
#endif // PYTHON_GRAPH_HPP
