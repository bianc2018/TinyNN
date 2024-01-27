// TinyNN.cpp: 定义应用程序的入口点。
//

#include "TinyNN.h"
#if 1
using namespace std;
double Fun(double x1)
{
    return 0.5 * x1+5;
}
int main()
{
    double flearing = 0.0001;
	nn::Network myNet(1, flearing);
	myNet.AddLayer(1);

    int max_loss_cnt=0;
	int e = 0;
	while (true)
	{
		++e;
		double x1 = rand()%10;
		std::vector<double> vecInput({ x1});
		std::vector<double> vecTarget({ Fun(x1)});
		std::vector<double> vecOutput;
		myNet.Forward(vecInput, vecOutput);
        double floss = myNet.GetLossError(vecTarget);
		printf("e:%d,lose:%lf,Output:%lf,Target:%lf\n", e, floss, vecOutput[0], vecTarget[0]);
		
        if (floss > 0.00000001)
        {
            max_loss_cnt = 0;
        }
        else
        {
            max_loss_cnt++;
            if (max_loss_cnt > 10)
                break;
        }
		myNet.Back(vecTarget);
	}
	
    for(int i=0;i<3;++i)
    {
        double x1 = rand() % 10;
        std::vector<double> vecInput({ x1 });
        std::vector<double> vecTarget({ Fun(x1) });
        std::vector<double> vecOutput;
        myNet.Forward(vecInput, vecOutput);
        double floss = myNet.GetLossError(vecTarget);
        printf("test:lose:%lf,Output:%lf,Target:%lf\n", floss, vecOutput[0], vecTarget[0]);
    }
	return 0;
}



#else
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

class ActivationFunction {
public:
    virtual double output(double input) const = 0;
    virtual double der(double input) const = 0;
};

class TanhActivation : public ActivationFunction {
public:
    double output(double input) const override {
        return std::tanh(input);
    }

    double der(double input) const override {
        double output = std::tanh(input);
        return 1 - output * output;
    }
};

class ReLUActivation : public ActivationFunction {
public:
    double output(double input) const override {
        return std::max(0.0, input);
    }

    double der(double input) const override {
        return input <= 0 ? 0 : 1;
    }
};

class SigmoidActivation : public ActivationFunction {
public:
    double output(double input) const override {
        return 1 / (1 + std::exp(-input));
    }

    double der(double input) const override {
        double output = this->output(input);
        return output * (1 - output);
    }
};

class LinearActivation : public ActivationFunction {
public:
    double output(double input) const override {
        return input;
    }

    double der(double input) const override {
        return 1;
    }
};

class Node {
public:
    std::string id;
    double bias = 0.1;
    double totalInput;
    double output;
    double outputDer = 0;
    double inputDer = 0;
    ActivationFunction* activation;

    Node(const std::string& id, ActivationFunction* activation, bool initZero = false)
        : id(id), activation(activation) {
        if (initZero) {
            bias = 0;
        }
    }

    double updateOutput() {
        totalInput = bias;
        for (const auto& link : inputLinks) {
            totalInput += link.weight * link.source->output;
        }
        output = activation->output(totalInput);
        return output;
    }
};

class ErrorFunction {
public:
    virtual double error(double output, double target) const = 0;
    virtual double der(double output, double target) const = 0;
};

class SquaredError : public ErrorFunction {
public:
    double error(double output, double target) const override {
        return 0.5 * std::pow(output - target, 2);
    }

    double der(double output, double target) const override {
        return output - target;
    }
};

class Link {
public:
    std::string id;
    Node* source;
    Node* dest;
    double weight;
    bool isDead = false;
    double errorDer = 0;
    RegularizationFunction* regularization;

    Link(Node* source, Node* dest, RegularizationFunction* regularization, bool initZero = false)
        : source(source), dest(dest), regularization(regularization) {
        id = source->id + "-" + dest->id;
        weight = initZero ? 0 : (std::rand() % 1000) / 500.0 - 1.0;
    }
};

class RegularizationFunction {
public:
    virtual double output(double weight) const = 0;
    virtual double der(double weight) const = 0;
};

class L1Regularization : public RegularizationFunction {
public:
    double output(double weight) const override {
        return std::abs(weight);
    }

    double der(double weight) const override {
        return weight < 0 ? -1 : (weight > 0 ? 1 : 0);
    }
};

class L2Regularization : public RegularizationFunction {
public:
    double output(double weight) const override {
        return 0.5 * weight * weight;
    }

    double der(double weight) const override {
        return weight;
    }
};

class NeuralNetwork {
public:
    std::vector<std::vector<Node>> network;
    std::vector<ErrorFunction*> errorFunctions;
    std::vector<RegularizationFunction*> regularizationFunctions;

    NeuralNetwork(const std::vector<int>& networkShape, ActivationFunction* activation,
        ActivationFunction* outputActivation, RegularizationFunction* regularization,
        const std::vector<std::string>& inputIds, bool initZero = false) {
        buildNetwork(networkShape, activation, outputActivation, regularization, inputIds, initZero);
    }

    void buildNetwork(const std::vector<int>& networkShape, ActivationFunction* activation,
        ActivationFunction* outputActivation, RegularizationFunction* regularization,
        const std::vector<std::string>& inputIds, bool initZero = false) {
        int numLayers = networkShape.size();
        int id = 1;
        network.clear();

        for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            bool isOutputLayer = layerIdx == numLayers - 1;
            bool isInputLayer = layerIdx == 0;
            std::vector<Node> currentLayer;
            network.push_back(currentLayer);
            int numNodes = networkShape[layerIdx];

            for (int i = 0; i < numNodes; i++) {
                std::string nodeId;
                if (isInputLayer) {
                    nodeId = inputIds[i];
                }
                else {
                    nodeId = std::to_string(id++);
                }

                Node node(nodeId, isOutputLayer ? outputActivation : activation, initZero);
                currentLayer.push_back(node);

                if (layerIdx >= 1) {
                    for (Node& prevNode : network[layerIdx - 1]) {
                        Link link(&prevNode, &node, regularization, initZero);
                        prevNode.outputs.push_back(link);
                        node.inputLinks.push_back(link);
                    }
                }
            }
        }
    }

    double forwardProp(const std::vector<double>& inputs) {
        std::vector<Node>& inputLayer = network[0];
        if (inputs.size() != inputLayer.size()) {
            throw std::runtime_error("The number of inputs must match the number of nodes in the input layer");
        }

        for (size_t i = 0; i < inputLayer.size(); i++) {
            Node& node = inputLayer[i];
            node.output = inputs[i];
        }

        for (size_t layerIdx = 1; layerIdx < network.size(); layerIdx++) {
            for (size_t i = 0; i < network[layerIdx].size(); i++) {
                Node& node = network[layerIdx][i];
                node.updateOutput();
            }
        }

        return network.back().front().output;
    }

    void backProp(double target) {
        Node& outputNode = network.back().front();
        outputNode.outputDer = errorFunctions.front()->der(outputNode.output, target);

        for (int layerIdx = network.size() - 1; layerIdx >= 1; layerIdx--) {
            for (Node& node : network[layerIdx]) {
                node.inputDer = node.outputDer * node.activation->der(node.totalInput);
                node.accInputDer += node.inputDer;
                node.numAccumulatedDers++;
            }

            for (Node& node : network[layerIdx]) {
                for (Link& link : node.inputLinks) {
                    if (!link.isDead) {
                        link.errorDer = node.inputDer * link.source->output;
                        link.accErrorDer += link.errorDer;
                        link.numAccumulatedDers++;
                    }
                }
            }

            if (layerIdx == 1) {
                continue;
            }

            for (Node& node : network[layerIdx - 1]) {
                node.outputDer = 0;
                for (Link& output : node.outputs) {
                    node.outputDer += output.weight * output.dest->inputDer;
                }
            }
        }
    }

    void updateWeights(double learningRate, double regularizationRate) {
        for (size_t layerIdx = 1; layerIdx < network.size(); layerIdx++) {
            for (Node& node : network[layerIdx]) {
                if (node.numAccumulatedDers > 0) {
                    node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
                    node.accInputDer = 0;
                    node.numAccumulatedDers = 0;
                }

                for (Link& link : node.inputLinks) {
                    if (!link.isDead && link.numAccumulatedDers > 0) {
                        link.weight = link.weight - (learningRate / link.numAccumulatedDers) * link.accErrorDer;

                        double newLinkWeight = link.weight - (learningRate * regularizationRate) * link.regularization->der(link.weight);

                        if (link.regularization == &L1Regularization && link.weight * newLinkWeight < 0) {
                            link.weight = 0;
                            link.isDead = true;
                        }
                        else {
                            link.weight = newLinkWeight;
                        }

                        link.accErrorDer = 0;
                        link.numAccumulatedDers = 0;
                    }
                }
            }
        }
    }

    void forEachNode(bool ignoreInputs, std::function<void(Node&)> accessor) {
        for (size_t layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.size(); layerIdx++) {
            for (Node& node : network[layerIdx]) {
                accessor(node);
            }
        }
    }

    Node& getOutputNode() {
        return network.back().front();
    }

    ~NeuralNetwork() {
        for (std::vector<Node>& layer : network) {
            layer.clear();
        }
        network.clear();
    }
};

int main() {
    // Example usage
    std::vector<int> networkShape = { 3, 4, 2, 1 };
    TanhActivation activation;
    LinearActivation outputActivation;
    L2Regularization regularization;
    std::vector<std::string> inputIds = { "Input1", "Input2", "Input3" };
    NeuralNetwork neuralNetwork(networkShape, &activation, &outputActivation, &regularization, inputIds);
}
#endif