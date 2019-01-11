#ifndef DIGIT_HPP
#define DIGIT_HPP
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <random>

namespace digitrecognition {

	class Neuron {
	public:
		Neuron(int inputs, int random_range, float bias);
		void activation();

		std::vector<float> values;
		std::vector<float> weights;
		float bias;
		float delta;
		std::vector<float> epoch_avg;

		float output; 
	};

	class Layer {
	public:
		std::vector<Neuron*> neurons;
	};

	class Network {
	public:
		Network(int n_inputs, int n_hidden, float alpha, int random_range, float bias);
		void train(FILE* image, std::ifstream& labels);
		void read(FILE* file);
		void calculateCost(std::ifstream&  labels);
		void backpropagate();

		void test(FILE* image, std::ifstream& labels);
		void classify(std::ifstream& labels);
		void printResults(int test_examples, int hidden);

		int n_inputs;
		int n_hidden;
		int n_output = 5;

		float alpha;
		int random_range;
		float cost;

		float prev_squared_errors;
		float mean_squared_error;
		int trained_examples = 0;
		bool done_training = false;

		Layer hiddenLayer;
		Layer outputLayer;

		std::vector<float> inputVec;
		std::vector<int> correctOutputs;
		std::vector<float> recent_mean_sqaured_errors;

		std::vector<std::vector<int>> confusionMatrix;
	};

	float sigmoid(float x);


}
#endif