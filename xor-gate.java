//my first xor gate nerual net try and its perfectly working with 0.001 error 

package xor;

import java.util.Random;

public class XorNeuralNet {
	double inputs[] = new double[2];

	double inputWeights[] = new double[6];
	double outputWeights[] = new double[3];
	double y3, y4, y6,y5;
	double theta3, theta4, theta6,theta5;
	double error;
	double sum3, sum4, sum6,sum5;
	double delta6, delta3, delta4,delta5;
	double momentum = 0;

	public XorNeuralNet() {
		inputWeights[0] = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		inputWeights[1] = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		inputWeights[2] =new Random().nextDouble();// (new Random().nextDouble() - 0.5) / 2.0;
		inputWeights[3] = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		inputWeights[4] =new Random().nextDouble();// (new Random().nextDouble() - 0.5) / 2.0;
		inputWeights[5] = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
	
		outputWeights[0] =new Random().nextDouble();// (new Random().nextDouble() - 0.5) / 2.0;
		outputWeights[1] = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		outputWeights[2] = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		
		/*theta3 = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		theta4 =new Random().nextDouble();// (new Random().nextDouble() - 0.5) / 2.0;
		theta6 = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		theta5 = new Random().nextDouble();//(new Random().nextDouble() - 0.5) / 2.0;
		*/
		delta6 = 0;
		delta3 = 0;
		delta4 = 0;
		delta5=0;
		theta3=0;
		theta4=0;
		theta5=0;
		theta6=0;
		
	}

	public void train(double a, double b, double desireOutput, double learningRate) {
		calculate(a, b);
		backpropagate(a, b, desireOutput, learningRate);
	}

	public void backpropagate(double a, double b, double desireOutput, double learningRate) {
		double delta5 = (desireOutput - y6) * sigmoidPrime(sum6);
		// double delta5 = (1 - y5)*y5*(desireOutput - y5);
		// double delta5 = (desireOutput - y5);
		// double delta5 = (desireOutput - y5) * tanhPrime(sum5);
		this.delta6 = delta5 * learningRate + momentum * this.delta6;
		// outputWeights[0] += learningRate * delta5 * y3;
		outputWeights[0] += this.delta6 * y3;
		// outputWeights[1] += learningRate * delta5 * y4;
		// theta5 = learningRate * delta5;

		outputWeights[1] += this.delta6 * y4;

		outputWeights[2] += this.delta6 * y5;
		theta6 = this.delta6;
		double delta3 = this.delta6 * outputWeights[0] * sigmoidPrime(sum3);
		double delta4 = this.delta6 * outputWeights[1] * sigmoidPrime(sum4);

		double delta6 = this.delta6 * outputWeights[2] * sigmoidPrime(sum5);
		/*outputWeights[0] += this.delta6 * y3;
		// outputWeights[1] += learningRate * delta5 * y4;
		// theta5 = learningRate * delta5;

		outputWeights[1] += this.delta6 * y4;

		outputWeights[2] += this.delta6 * y5;*/
		/*
		 * double delta3=delta5*outputWeights[0]*y3*(1-y3); double
		 * delta4=delta5*outputWeights[1]*y4*(1-y4);
		 */
		/*
		 * double delta3 = this.delta5 * outputWeights[0] * tanhPrime(sum3);
		 * double delta4 = this.delta5 * outputWeights[1] * tanhPrime(sum4);
		 */
		this.delta3 = delta3 * learningRate + momentum * this.delta3;
		this.delta4 = delta4 * learningRate + momentum * this.delta4;
		this.delta5 = delta6 * learningRate + momentum * this.delta5;

		/*
		 * inputWeights[0] += learningRate * delta3 * a; inputWeights[1] +=
		 * learningRate * delta3 * b; inputWeights[2] += learningRate * delta4 *
		 * a; inputWeights[3] += learningRate * delta4 * b; theta3 =
		 * learningRate * delta3; theta4 = learningRate * delta4;
		 */
		inputWeights[0] += this.delta3 * a;
		inputWeights[1] += this.delta3 * b;
		inputWeights[2] += this.delta4 * a;
		inputWeights[3] += this.delta4 * b;

		inputWeights[4] += this.delta5 * a;
		inputWeights[5] += this.delta5 * b;
		theta3 = this.delta3;
		theta4 = this.delta4;
		theta5 = this.delta5;

		// regularization
		/*
		 * if (inputWeights[0] < 0.1) { inputWeights[0] = 0; } else if
		 * (inputWeights[0] > 0.9) { inputWeights[0] = 0.8; } if
		 * (inputWeights[2] < 0.1) { inputWeights[2] = 0; } else if
		 * (inputWeights[2] > 0.9) { inputWeights[2] = 0.8; } if
		 * (inputWeights[1] < 0.1) { inputWeights[1] = 0; } else if
		 * (inputWeights[3] > 0.9) { inputWeights[1] = 0.8; } if
		 * (inputWeights[3] < 0.1) { inputWeights[3] = 0; } else if
		 * (inputWeights[3] > 0.9) { inputWeights[3] = 0.8; }
		 */
		/*
		 * if(outputWeights[0]<0.1){ outputWeights[0]=0.1; } else
		 * if(outputWeights[0]>0.9){ outputWeights[0]=0.9; }
		 * if(outputWeights[1]<0.1){ outputWeights[1]=0.1; } else
		 * if(outputWeights[1]>0.9){ outputWeights[1]=0.9; }
		 */
	}

	public void error(double desireOutput) {

		error = (desireOutput - y6) * (desireOutput - y6) / 2;
		System.out.println("desire ouput:" + desireOutput + ",  neural output:" + y6 + " ,error calculated:" + error);
	}

	public void calculate(double a, double b) {
		sum3 = a * inputWeights[0] + b * inputWeights[1] + theta3;
		y3 = sigmoid(sum3);
		// y3 = Math.tanh(sum3);
		sum4 = a * inputWeights[2] + b * inputWeights[3] + theta4;
		y4 = sigmoid(sum4);
		// y4 = Math.tanh(sum4);

		sum5 = a * inputWeights[4] + b * inputWeights[5] + theta5;
		y5 = sigmoid(sum5);
		sum6 = y3 * outputWeights[0] + y4 * outputWeights[1] +y5*outputWeights[2]+ theta6;
		y6 = sigmoid(sum6);
		// y5 = Math.tanh(sum5);
	}

	public double sigmoid(double z) {
		return (double) (1.0 / (1 + Math.exp(-z)));

	}

	public double tanhPrime(double z) {
		double result = 1 - Math.pow(Math.tanh(z), 2);
		return result;
	}

	public double sigmoidPrime(double z) {
		/*
		 * double exp = (double) Math.exp(-z); return (double) (exp /
		 * Math.pow((1 + exp),2));
		 */
		return (sigmoid(z) * (1 - sigmoid(z)));
	}

	public static void main(String[] args) {
		XorNeuralNet xor = new XorNeuralNet();
		double learnRate = 1;
		/*
		 * float input[][]={{0,0},{1,0},{0,1},{1,1}}; float output[]={0,1,1,0};
		 */
		for (int i = 0; i < 1000; i++) {
			xor.train(0.0, 0.0, 0.0, learnRate);
			// xor.error(0.0);
			xor.train(0.0, 1.0, 1.0, learnRate);
			// xor.error(1.0);
			xor.train(1.0, 0.0, 1.0, learnRate);
			// xor.error(1.0);
			xor.train(1.0, 1.0, 0.0, learnRate);
			// xor.error(0.0);

		}
		
		xor.calculate(0, 0);
		xor.error(0.0);
		xor.calculate(1, 0);
		xor.error(1.0);
		xor.calculate(0, 1);
		xor.error(1.0);
		xor.calculate(1, 1);
		xor.error(0.0);
		
		
	}
}
