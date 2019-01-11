// Alexander Ngo
// AI Project: Handwritten Digit Recognition
// Two-Layer Fully-Connected Perception Neural Network for recognizing digits from 0 to 4

#include "digit.h"
using namespace std;
using namespace digitrecognition;

int main() {

	//dataset
	int n_train = 28038;
	int n_test = 2561;

	//layers
	int n_inputs = 784; // 28x28 pixels
	int n_hidden = 50;

	//caculation related variables
	float alpha = 0.1f;
	int random_range = 1;
	float bias = -1.0f;

	Network* network = new Network(n_inputs, n_hidden, alpha, random_range, bias);

	char trainFile[20] = "train_images.raw";
	char testFile[20] = "test_images.raw";
	FILE *fp1;
	ifstream trainLabel("train_labels.txt");
	FILE *fp2;
	ifstream testLabel("test_labels.txt");
	fp1 = fopen(trainFile, "rb");
	fp2 = fopen(testFile, "rb");
	
	//train the network
	int i = 0;
	while (i < n_train && !network->done_training) {
		network->train(fp1, trainLabel);
		i++;
	}

	cout << endl;
	
	//test the network
	i = 0;
	while (i < n_test) {
		network->test(fp2, testLabel);
		i++;
	}
	
	network->printResults(n_test, n_hidden);

	fclose(fp1);
	trainLabel.close();
	fclose(fp2);
	testLabel.close();
	return 0;
}