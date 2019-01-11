#include "digit.h"
using namespace std;

namespace digitrecognition {

	ofstream outfile("Results.txt");

	float sigmoid(float x) {
		// 1/ (1 + e^-x)
		float result = 1 / (1 + (exp(-x)));
		return result;
	}

	Network::Network(int n_inputs, int n_hidden, float alpha, int random_range, float bias) : n_inputs(n_inputs), n_hidden(n_hidden), alpha(alpha), random_range(random_range) {
		for (int i = 0; i < n_hidden; i++) {
			Neuron* np = new Neuron(n_inputs, random_range, bias);
			hiddenLayer.neurons.push_back(np);
		}
		for (int i = 0; i < 5; i++) {
			Neuron* np = new Neuron(n_hidden, random_range, bias);
			outputLayer.neurons.push_back(np);
		}
		inputVec.insert(inputVec.end(), n_inputs, 0.0f);
		correctOutputs.insert(correctOutputs.end(), 5, 0);

		vector<int> trueClass{ 0,0,0,0,0 };
		confusionMatrix.push_back(trueClass);
		confusionMatrix.push_back(trueClass);
		confusionMatrix.push_back(trueClass);
		confusionMatrix.push_back(trueClass);
		confusionMatrix.push_back(trueClass);
	}

	Neuron::Neuron(int inputs, int random_range, float bias) : bias(bias) {

		//Initializing random seed
		std::random_device rd;
		std::mt19937 gen(rd());

		for (int i = 0; i < inputs; i++) {

			//Generate a random weight between negative and positive random_range (to the hundredths place)
			std::uniform_int_distribution<> rnum(random_range*-100, random_range * 100);
			float weight = (float)rnum(gen);
			weight /= 100.0f;

			weights.push_back(weight);
			values.push_back(0.0f);
			epoch_avg.push_back(0.0f);
		}
	}

	void Neuron::activation() {
		float weightedSum = 0;

		//First compute weighted sum
		for (size_t i = 0; i < values.size(); i++) {
			weightedSum += weights[i] * values[i];
		}
		weightedSum += bias;
		//The compute output using weighted sum and sigmoid function
		output = sigmoid(weightedSum);
	}

	void Network::read(FILE* file) {
		short ch;
		int i = 0;
		while (i != n_inputs) {
			ch = getc(file);
			inputVec[i] = ((float)((unsigned int)ch)) / 255.0f;
			i++;
		}
		

		//Feed data from inputVector into each hiddenLayer neuron
		for (int i = 0; i < n_hidden; i++) {

			for (int j = 0; j < n_inputs; j++) {
				hiddenLayer.neurons[i]->values[j] = inputVec[j];
			}
		}

	}

	void Network::calculateCost(ifstream& labels) {

		labels >> correctOutputs[0] >> correctOutputs[1] >> correctOutputs[2] >> correctOutputs[3] >> correctOutputs[4];
		cost = 0.0f;
		for (int i = 0; i < 5; i++) {
			cost += 0.5f * pow((float)correctOutputs[i] - outputLayer.neurons[i]->output, 2);
		}

		prev_squared_errors += cost;
		trained_examples++;
		mean_squared_error = prev_squared_errors/ trained_examples;

		//Add the mean sqaured error to the vector of recent mean squared errors and check if it changed much at all from the one 300 iterations ago
		//If there's not much of a difference, we can stop training and go to testing
		if (recent_mean_sqaured_errors.size() < 300) {
			recent_mean_sqaured_errors.push_back(mean_squared_error);
		}
		else {
			//Check if we should stop training
			//Not allowed to start checking until at least 1000 exampels have been used to train
			if (recent_mean_sqaured_errors[0] - mean_squared_error < 0.01f && trained_examples > 1000) {
				done_training = true;
			}
			//Then move all iterations down the vector
			recent_mean_sqaured_errors.erase(recent_mean_sqaured_errors.begin());
			recent_mean_sqaured_errors.push_back(mean_squared_error);
		}
	}

	void Network::backpropagate() {

		//Start with output layer
		for (int i = 0; i < 5; i++) {

			//Calculate delta of output to hidden layer
			float weightedSum = 0.0f;

			for (int j = 0; j < n_hidden; j++) {
				weightedSum += outputLayer.neurons[i]->weights[j] * outputLayer.neurons[i]->values[j];
			}
			float derived_sigmoid = sigmoid(weightedSum) * (1 - sigmoid(weightedSum));
			outputLayer.neurons[i]->delta = ((float)correctOutputs[i] - outputLayer.neurons[i]->output) * derived_sigmoid;

			for (int j = 0; j < n_hidden; j++) {
				outputLayer.neurons[i]->weights[j] += alpha * outputLayer.neurons[i]->delta * hiddenLayer.neurons[j]->output;
			}
		}


		//Backpropagate to hidden layer
		for (int i = 0; i < n_hidden; i++) {

			float weightedSum = 0.0f;
			
			//Calculate derived_sigmoid for delta value
			for (int j = 0; j < n_inputs; j++) {
				
				weightedSum += hiddenLayer.neurons[i]->weights[j] * hiddenLayer.neurons[i]->values[j];
			}
			float derived_sigmoid = sigmoid(weightedSum) * (1 - sigmoid(weightedSum));

			float summation = 0.0f;
			//Calculate summation of (weights to output layer * corresponding output delta)
			for (int k = 0; k < 5; k++) {
				summation += outputLayer.neurons[k]->weights[i] * outputLayer.neurons[k]->delta;
			}

			//Combined above two parts for delta
			hiddenLayer.neurons[i]->delta = derived_sigmoid * summation;

			for (int j = 0; j < n_inputs; j++) {
				hiddenLayer.neurons[i]->weights[j] += alpha * hiddenLayer.neurons[i]->delta * inputVec[j];
			}
		}
	}

	void Network::train(FILE* image, std::ifstream& labels) {
		read(image);
		
		//Call activation function
		for (int i = 0; i < n_hidden; i++) {
			hiddenLayer.neurons[i]->activation();
		}
		
		//Feed hidden layer into output layer and call activation function
		for (int i = 0; i < n_output; i++) {
			  
			for (int j = 0; j < n_hidden; j++) {
				outputLayer.neurons[i]->values[j] = hiddenLayer.neurons[j]->output;
			}
			outputLayer.neurons[i]->activation();
		}
		
		calculateCost(labels);
		backpropagate();
	}

	void Network::classify(std::ifstream& labels) {
		labels >> correctOutputs[0] >> correctOutputs[1] >> correctOutputs[2] >> correctOutputs[3] >> correctOutputs[4];

		int trueClass;
		for (size_t i = 0; i < correctOutputs.size(); i++) {
			if (correctOutputs[i] == 1) {
				trueClass = i;
			}
		}

		int outputClass;
		float max = 0.0f;
		for (size_t i = 0; i < outputLayer.neurons.size(); i++) {
			if (outputLayer.neurons[i]->output > max) {
				max = outputLayer.neurons[i]->output;
				outputClass = i;
			}
		}
		
		//Put this classification in the confusion matrix
		confusionMatrix[trueClass][outputClass]++;
	}

	void Network::test(FILE* image, std::ifstream& labels) {
		read(image);

		//Call activation function
		for (int i = 0; i < n_hidden; i++) {
			hiddenLayer.neurons[i]->activation();
		}

		//Feed hidden layer into output layer and call activation function
		for (int i = 0; i < n_output; i++) {

			for (int j = 0; j < n_hidden; j++) {
				outputLayer.neurons[i]->values[j] = hiddenLayer.neurons[j]->output;
			}
			outputLayer.neurons[i]->activation();
		}

		//Check if network classifies image correctly
		classify(labels);
	}

	void Network::printResults(int test_examples, int hidden) {
		//Print the confusion matrix
		outfile << "    " << "0   " << "1   " << "2   " << "3   " << "4   " << endl;

		for (size_t i = 0; i < confusionMatrix.size(); i++) {

			outfile << i << "   ";

			for (size_t j = 0; j < confusionMatrix[i].size(); j++) {
				if (confusionMatrix[i][j] < 10) {
					outfile << confusionMatrix[i][j] << "   ";
				}
				else if (confusionMatrix[i][j] < 100) {
					outfile << confusionMatrix[i][j] << "  ";
				}
				else {
					outfile << confusionMatrix[i][j] << " ";
				}
			}

			outfile << endl;
		}


		//Print accuracy
		int accuracy = (int)( ((float)(confusionMatrix[0][0] + confusionMatrix[1][1] + confusionMatrix[2][2] + confusionMatrix[3][3] + confusionMatrix[4][4]) / (float)test_examples) * 100);
		outfile << endl << "The overall accuracy is: " << accuracy << "%" << endl;
		outfile << endl << hidden << " neurons were used in the hidden layer." << endl;
		outfile << trained_examples << " images were used to train this neural network." << endl; 
		outfile << test_examples << " images were used to test his neural network." << endl;
	}

}