# Quantized LSTM Model for FPGA-based Edge Computation on a Vibration Sensor Package

## Introduction

In the evolving field of Internet of Things (IoT) and edge computing, the ability to process data at the source is becoming increasingly crucial. This project revolves around the deployment of a Quantized Long Short-Term Memory (LSTM) model for edge computation, specifically tailored for FPGA (Field-Programmable Gate Array) hardware, in managing and interpreting data from a Vibration Sensor Package.

### Why FPGA for Edge Computing?

Edge computing demands low-latency, power-efficiency, and real-time data processing capabilities in various applications, such as predictive maintenance, anomaly detection, and real-time monitoring in industrial IoT. FPGAs, due to their inherent characteristics, emerge as a viable solution for such edge computing tasks:

1. **Low Latency**: FPGAs facilitate parallel processing which substantially reduces the data processing time, ensuring low-latency performance which is pivotal in time-sensitive applications.
   
2. **Power Efficiency**: Unlike traditional CPUs and GPUs, FPGAs can be optimized for specific tasks, like running neural networks, which significantly reduces power consumption making them suitable for edge devices that often run on battery power.
   
3. **Flexibility and Reconfigurability**: FPGAs can be reprogrammed and reconfigured to adapt to different applications and algorithms, making them a flexible hardware choice for evolving edge computing needs.
   
4. **Security**: FPGAs offer secure data processing at the edge, reducing the need to transmit sensitive data to the cloud, and thus mitigating potential security risks.
   
By employing a quantized LSTM model, the project further optimizes the resource utilization and power consumption, ensuring a lightweight and efficient model deployment on FPGAs. The quantization reduces the precision of the weights, thereby reducing memory requirements and computational complexity, which is particularly beneficial for resource-constrained edge devices.

## Getting Started

### Prerequisites and Installation

Ensure that you have the following dependencies installed to run the code. If not, follow the subsequent steps to set up your environment:

#### Dependencies:
- [Python](https://www.python.org/downloads/)
- [NumPy](https://numpy.org/install/)
- [TensorFlow](https://www.tensorflow.org/install)
- [ONNX](https://onnx.ai/get-started.html)
- [PyTorch (Torch)](https://pytorch.org/get-started/locally/)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)
- [tqdm](https://tqdm.github.io/)
- [FINN](https://xilinx.github.io/finn/)
- [HLS4ML](https://fastmachinelearning.org/hls4ml/) (for translating trained models into FPGA firmware)



