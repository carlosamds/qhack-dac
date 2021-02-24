#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

from constants import N_LAYERS, N_WIRES

dev = qml.device("default.qubit", wires=N_WIRES)

def layer(theta):
    '''A layer of fixed quantum gates.'''
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    for j in range(len(theta)):
        qml.Rot(theta[j], np.pi/2, np.pi, wires=j)


def variational_circuit(theta, x):
    '''A layered variational circuit'''
    state_preparation(x)
    for i in range(N_LAYERS):
        layer(theta[i])


@qml.qnode(dev)
def qnode(theta, x):
    variational_circuit(theta, x)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_WIRES)]


def state_preparation(x):
    # x = 0 or 1 TODO
    for i in range(len(x)):
        qml.Rot(x[i]*np.pi, 0, x[i]*np.pi, wires=i)


def cost(qcircuit, params, xs, ys, actions):
    loss = 0.0
    for x, y, act in zip(xs, ys, actions):
        pred = qcircuit(params, x)[act]
        loss += (pred - y)**2
    return loss
