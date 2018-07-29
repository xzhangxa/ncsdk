
# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import networkx as nx
import pygraphviz as pgv
import os
import matplotlib.pyplot as plt
import tempfile
import networkx as nx
from mvnctools.Controllers.GraphUtils import buildGraph

CUR_DIR = os.path.dirname(__file__)

def drawIR(parsedLayers):
    """
        Draws two graphs from the parsedLayers object.
        - with tensors
        - without tensors.

        Currently writes in the same directory as this file.
        TODO: path arg.
        TODO: Issue with Conversion Layers.
    """

    # Create empty graph. We want to preserve the order of parents/children
    g = nx.OrderedMultiDiGraph()

    # Add transformations as nodes and connect them with blobs
    for layer in parsedLayers:

        for inputTensor in layer.getInputTensorNames():
            g.add_node("Tensor_" + inputTensor.stringifyName(), type="BLOB", ref=None)

        for outputTensor in layer.getOutputTensorNames():
            g.add_node("Tensor_" + outputTensor.stringifyName(), type="BLOB", ref=None)


    for layer in parsedLayers:
        g.add_node("Op_" + layer.getName().stringifyName(), type="OP", ref=layer)

        for inputTensor in layer.getInputTensorNames():
            g.add_edge("Tensor_" + inputTensor.stringifyName(), "Op_" + layer.getName().stringifyName())

        for outputTensor in layer.getOutputTensorNames():
            g.add_edge("Op_" + layer.getName().stringifyName(), "Tensor_" + outputTensor.stringifyName())

    g.graph['graph']={'rankdir':'TD'}
    g.graph['node'] = {'shape':'box'}
    g.graph['edges']={'arrowsize':'4.0'}

    A = nx.nx_agraph.to_agraph(g)
    A.layout('dot')
    A.draw(os.path.join(CUR_DIR, 'graph_with_tensors.png'))

    for n in g.node:
        if g.node[n]['type'] == 'BLOB':
            for parent in g.predecessors(n):
                g = nx.contracted_edge(g, (parent, n), self_loops=False)

    g.graph['graph']={'rankdir':'TD'}
    g.graph['node']={'shape':'box'}
    g.graph['edges']={'arrowsize':'4.0'}

    A = nx.nx_agraph.to_agraph(g)
    A.layout('dot')
    A.draw(os.path.join(CUR_DIR, 'graph_without_tensors.png'))

def drawIRNetworkX(g):
    """
        Draws the inbuilt visualization of networkx graphs
        TODO: path arg.
    """
    nx.draw(g)
    plt.show()


def drawGraph(G, path=None):
    """
        a coloured version of the graph from networkX
    """
    g = G.copy()

    g.graph['graph']={'rankdir':'TD'}
    g.graph['node']={'shape':'box'}
    g.graph['edges']={'arrowsize':'4.0'}

    for name in g:
        if g.node[name]['type'] == 'BLOB':
            g.node[name]['fillcolor']='yellow'
            g.node[name]['style']='filled'

    A = nx.nx_agraph.to_agraph(g)
    A.layout('dot')

    if path:
        filename = os.path.join(path)
        A.draw(os.path.join(path))
    else:
        fp, filename = tempfile.mkstemp(suffix='.png')
        A.draw(filename)

    print('Graph written to {}'.format(filename))

def drawGraphFromLayers(parsedLayers, path=None):
    """
    As above but with a parsedLayer input
    """
    g = buildGraph(parsedLayers)
    drawGraph(g, path)
