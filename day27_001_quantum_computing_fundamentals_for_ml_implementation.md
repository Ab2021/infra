# Day 27.1: Quantum Computing Fundamentals for ML - A Practical Introduction

## Introduction: The Dawn of a New Computing Paradigm

For decades, machine learning has been powered by classical computers, operating on bits that are either 0 or 1. However, as datasets grow and models become more complex, we are beginning to push the limits of classical computation. Enter **Quantum Computing**, a revolutionary paradigm that leverages the principles of quantum mechanics to process information in fundamentally new ways.

Instead of bits, quantum computers use **qubits**, which can exist in a state of **superposition** (both 0 and 1 at the same time) and can be linked together through **entanglement**. These properties unlock an exponentially larger computational space, offering the potential to solve certain problems that are intractable for even the most powerful classical supercomputers.

This guide will serve as a practical introduction to the core concepts of quantum computing that are most relevant for machine learning practitioners. We will explore the fundamental building blocks of quantum computation and, most importantly, write code to simulate these concepts, bridging the gap between abstract theory and practical implementation.

**Today's Learning Objectives:**

1.  **Understand the Qubit:** Grasp the concept of a qubit, its mathematical representation, and how it fundamentally differs from a classical bit.
2.  **Explore Quantum Principles:** Learn about superposition and entanglement, the two pillars of quantum computation, and see them in action.
3.  **Learn Quantum Gates:** Understand how quantum operations (gates) are used to manipulate qubits, analogous to logic gates in classical computing.
4.  **Build and Simulate a Quantum Circuit:** Use a modern quantum computing library (PennyLane) to construct and execute a simple quantum circuit from scratch.
5.  **Encode Classical Data:** Discover techniques for representing classical data in a quantum state, a critical first step for any quantum machine learning task.

---

## Part 1: The Qubit - The Heart of the Quantum Computer

A classical bit is simple: it's either a 0 or a 1. A **qubit**, however, is far more expressive. It is a two-level quantum mechanical system, and its state can be a **linear combination** of its two basis states, |0⟩ and |1⟩ (pronounced "ket 0" and "ket 1").

The state of a qubit, |ψ⟩, is represented as:
|ψ⟩ = α|0⟩ + β|1⟩

Where:
*   **|0⟩** and **|1⟩** are the basis vectors, corresponding to the classical 0 and 1 states. Mathematically, they are represented as column vectors:
    |0⟩ = [1, 0]ᵀ
    |1⟩ = [0, 1]ᵀ
*   **α** and **β** are complex numbers called **probability amplitudes**.
*   The probability of measuring the qubit in the |0⟩ state is |α|², and the probability of measuring it in the |1⟩ state is |β|².
*   The sum of these probabilities must equal 1: |α|² + |β|² = 1.

This ability to be in a combination of states simultaneously is called **superposition**.

### 1.1. Simulating a Qubit with PennyLane

[PennyLane](https://pennylane.ai/) is a powerful Python library that allows us to simulate quantum circuits and, crucially, integrate them with PyTorch and other ML frameworks.

First, let's set up our environment and a "device" to run our simulation.

```python
# Ensure you have PennyLane installed: pip install pennylane
import pennylane as qml
from pennylane import numpy as np

print("--- Part 1: The Qubit ---")

# A "device" is the computational engine for our simulation.
# "default.qubit" is PennyLane's standard, high-performance simulator.
# `wires=1` means we are creating a device for a single qubit.
dev1 = qml.device("default.qubit", wires=1)

# A "QNode" binds a quantum function (our circuit) to a device.
@qml.qnode(dev1)
def create_qubit_state(alpha, beta):
    # We can initialize a qubit to a specific state.
    # Note: PennyLane requires the state vector to be normalized.
    state = np.array([alpha, beta])
    qml.QubitStateVector(state, wires=0)
    # `qml.expval(qml.PauliZ(0))` returns the expectation value of the Z operator,
    # which can distinguish between the |0> and |1> states.
    return qml.expval(qml.PauliZ(0))

# --- Example 1: A qubit in the |0> state ---
# |ψ⟩ = 1|0⟩ + 0|1⟩
alpha_0, beta_0 = 1.0, 0.0
exp_val_0 = create_qubit_state(alpha_0, beta_0)
print(f"Qubit state: {alpha_0}|0> + {beta_0}|1>")
print(f"Expectation value (Z): {exp_val_0:.2f}") # Should be 1 for |0>

# --- Example 2: A qubit in the |1> state ---
# |ψ⟩ = 0|0⟩ + 1|1⟩
alpha_1, beta_1 = 0.0, 1.0
exp_val_1 = create_qubit_state(alpha_1, beta_1)
print(f"\nQubit state: {alpha_1}|0> + {beta_1}|1>")
print(f"Expectation value (Z): {exp_val_1:.2f}") # Should be -1 for |1>

# --- Example 3: A qubit in superposition ---
# |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
alpha_s, beta_s = 1/np.sqrt(2), 1/np.sqrt(2)
exp_val_s = create_qubit_state(alpha_s, beta_s)
print(f"\nQubit state: {alpha_s:.2f}|0> + {beta_s:.2f}|1>")
print(f"Expectation value (Z): {exp_val_s:.2f}") # Should be 0, indicating an equal mix
```

---

## Part 2: Quantum Gates - Manipulating Qubits

If qubits are the quantum equivalent of bits, then **quantum gates** are the equivalent of classical logic gates (AND, OR, NOT). Gates are operations that manipulate the state of qubits. Mathematically, they are represented by unitary matrices.

### 2.1. Single-Qubit Gates

These gates act on a single qubit.

*   **Pauli-X Gate (Quantum NOT Gate):** Flips the |0⟩ state to |1⟩ and vice-versa.
    X = [[0, 1], [1, 0]]
*   **Hadamard Gate (H):** The "superposition gate." It transforms a qubit from a basis state (|0⟩ or |1⟩) into an equal superposition of both.
    H = (1/√2) * [[1, 1], [1, -1]]
*   **Pauli-Z Gate:** Introduces a phase shift. It leaves |0⟩ unchanged and flips the sign of |1⟩ to -|1⟩.
    Z = [[1, 0], [0, -1]]

### 2.2. Multi-Qubit Gates and Entanglement

These gates act on two or more qubits, and they are essential for creating **entanglement**. Entanglement is a uniquely quantum phenomenon where the states of multiple qubits become correlated in such a way that their fates are intertwined, no matter how far apart they are.

*   **Controlled-NOT Gate (CNOT):** A two-qubit gate. It flips the second qubit (the *target*) if and only if the first qubit (the *control*) is in the |1⟩ state. This is the key gate for creating entanglement.

### 2.3. Code Example: Applying Gates and Creating a Bell State

A **Bell state** is the simplest example of an entangled two-qubit state. We can create one by putting one qubit in superposition (with a Hadamard gate) and then applying a CNOT gate.

```python
print("\n--- Part 2: Quantum Gates and Entanglement ---")

# Use a device with 2 qubits
dev2 = qml.device("default.qubit", wires=2)

@qml.qnode(dev2)
def create_bell_state():
    # 1. Start with both qubits in the |0> state: |00>
    
    # 2. Apply a Hadamard gate to the first qubit (wire 0).
    # This puts it in superposition: (1/√2)(|0> + |1>)|0> = (1/√2)(|00> + |10>)
    qml.Hadamard(wires=0)
    
    # 3. Apply a CNOT gate.
    # Control qubit is wire 0, target qubit is wire 1.
    # If control is |0>, target is unchanged.
    # If control is |1>, target is flipped.
    # Result: (1/√2)(|00> + |11>) -> This is the Bell State!
    qml.CNOT(wires=[0, 1])
    
    # We return the probability of each of the 4 possible outcomes: 00, 01, 10, 11
    return qml.probs(wires=[0, 1])

# Execute the circuit and print the probabilities
probs = create_bell_state()
print("Circuit to create a Bell State (|Φ+⟩):")
print("1. H gate on qubit 0")
print("2. CNOT gate with control=0, target=1")
print(f"\nProbabilities of measuring each state:")
print(f"  P(00): {probs[0]:.2f}") # Should be 0.5
print(f"  P(01): {probs[1]:.2f}") # Should be 0.0
print(f"  P(10): {probs[2]:.2f}") # Should be 0.0
print(f"  P(11): {probs[3]:.2f}") # Should be 0.5

# The result shows that we will only ever measure 00 or 11.
# The states of the two qubits are perfectly correlated. This is entanglement.
```

---

## Part 3: Building a Complete Quantum Circuit

A quantum algorithm is implemented as a **quantum circuit**. This is a sequence of quantum gates applied to a set of qubits, typically followed by a measurement.

A typical circuit structure involves three main stages:
1.  **State Preparation / Data Encoding:** Prepare the initial state of the qubits. In QML, this is where you encode your classical data.
2.  **Model / Unitary Evolution:** Apply a sequence of gates that constitutes your model. This is often a parameterized part of the circuit that can be trained.
3.  **Measurement:** Extract information from the circuit, collapsing the quantum state into a classical outcome.

### 3.1. Code Example: A Simple Parameterized Circuit

Let's build a circuit that takes a classical input value, uses it to rotate a qubit, and then measures the outcome. This is the foundation of many Variational Quantum Algorithms.

```python
print("\n--- Part 3: A Simple Parameterized Circuit ---")

dev3 = qml.device("default.qubit", wires=1)

@qml.qnode(dev3)
def simple_parameterized_circuit(x):
    # Stage 1: State Preparation (already in |0>)
    
    # Stage 2: Model
    # qml.RX is a rotation around the X-axis of the Bloch sphere.
    # We use our classical input `x` as the angle for the rotation.
    # This is a simple way to "parameterize" the circuit.
    qml.RX(x, wires=0)
    
    # Stage 3: Measurement
    # Return the expectation value of the Pauli-Z operator.
    return qml.expval(qml.PauliZ(0))

# --- Test the circuit with different inputs ---
input_0 = 0.0
output_0 = simple_parameterized_circuit(input_0)
print(f"Input: {input_0:.2f}, Output (expval Z): {output_0:.2f}") # RX(0) does nothing, so state is |0>, expval is 1

input_half_pi = np.pi / 2
output_half_pi = simple_parameterized_circuit(input_half_pi)
print(f"Input: {input_half_pi:.2f}, Output (expval Z): {output_half_pi:.2f}") # RX(pi/2) creates a state on the equator, expval is 0

input_pi = np.pi
output_pi = simple_parameterized_circuit(input_pi)
print(f"Input: {input_pi:.2f}, Output (expval Z): {output_pi:.2f}") # RX(pi) is a NOT gate, state is |1>, expval is -1
```

This simple example shows how a classical parameter can influence the outcome of a quantum circuit, which is the core idea behind training quantum models.

---

## Part 4: Encoding Classical Data into Quantum States

To perform machine learning with a quantum computer, we must first find a way to represent our classical data (e.g., a feature vector from a dataset) in the quantum domain. This is called **data encoding** or **quantum embedding**.

There are several strategies, each with its own trade-offs.

### 4.1. Basis Encoding
The most straightforward method. A classical binary string like `101` is directly mapped to the corresponding qubit basis state |101⟩. This is simple but not very powerful, as it doesn't leverage superposition.

### 4.2. Angle Encoding
Here, we encode classical data into the rotation angles of single-qubit gates. For a feature vector `x = [x₁, x₂, ..., xₙ]`, we can use `n` qubits and apply a rotation `RX(xᵢ)` to the i-th qubit. We used this in our parameterized circuit above.

### 4.3. Amplitude Encoding
This is a very powerful and efficient technique. It encodes a classical N-dimensional vector `x` into the *amplitudes* of a quantum state of just `log₂(N)` qubits. For example, a 4-dimensional vector `[x₀, x₁, x₂, x₃]` can be encoded into the amplitudes of a 2-qubit state:
|ψ⟩ = x₀|00⟩ + x₁|01⟩ + x₂|10⟩ + x₃|11⟩
This requires the classical vector to be normalized (sum of squares of elements is 1).

### 4.4. Code Example: Angle Encoding a Feature Vector

Let's encode a 2D classical vector into the state of a 2-qubit system using angle encoding.

```python
print("\n--- Part 4: Angle Encoding ---")

dev4 = qml.device("default.qubit", wires=2)

@qml.qnode(dev4)
def angle_encoding_circuit(x):
    # x is a classical 2D vector, e.g., [0.5, 1.2]
    
    # Encode the first feature into the first qubit
    qml.RX(x[0], wires=0)
    
    # Encode the second feature into the second qubit
    qml.RX(x[1], wires=1)
    
    # Return the joint probability of the two qubits
    return qml.probs(wires=[0, 1])

# A sample classical feature vector
feature_vector = np.array([np.pi / 2, np.pi])
print(f"Encoding classical vector: {feature_vector}")

# Run the encoding circuit
encoded_probs = angle_encoding_circuit(feature_vector)

print("\nResulting probabilities after encoding:")
print(f"  P(00): {encoded_probs[0]:.2f}")
print(f"  P(01): {encoded_probs[1]:.2f}")
print(f"  P(10): {encoded_probs[2]:.2f}")
print(f"  P(11): {encoded_probs[3]:.2f}")
```

## Conclusion

In this guide, we have taken our first practical steps into the world of quantum computing for machine learning. We've moved from the abstract concepts of qubits and superposition to writing concrete Python code to simulate them. We have seen that the core ideas, while non-intuitive, can be systematically implemented and explored using modern software tools like PennyLane.

**Key Takeaways:**

1.  **Qubits are Vectors:** The state of a qubit is a vector, and its ability to be in a superposition of states is the source of its power.
2.  **Gates are Matrices:** Quantum gates are matrix operations that rotate these state vectors in the computational space.
3.  **Entanglement Creates Correlation:** Gates like CNOT can create deep correlations between qubits, a resource that has no classical equivalent.
4.  **Circuits are Algorithms:** Quantum algorithms are expressed as circuits, and we can build and simulate them easily in Python.
5.  **Data Encoding is the Bridge:** To apply quantum computing to ML, we must first have a strategy to embed our classical data into the quantum state space.

With these fundamentals in hand, we are now ready to explore how to build and train actual quantum machine learning models.

## Self-Assessment Questions

1.  **Superposition:** If a qubit is in the state `(1/√2)|0⟩ + (1/√2)|1⟩`, what is the probability of measuring it in the |0⟩ state?
2.  **Hadamard Gate:** What is the effect of applying a Hadamard gate to a qubit that is already in the |1⟩ state?
3.  **Entanglement:** In our Bell state example, if you measure the first qubit and find it to be in the |1⟩ state, what state will you instantly know the second qubit is in, and why?
4.  **Parameterized Circuits:** In the context of QML, why is it important for a quantum circuit to be "parameterized"?
5.  **Amplitude Encoding:** How many qubits would you need to encode a feature vector with 16 elements using amplitude encoding?
