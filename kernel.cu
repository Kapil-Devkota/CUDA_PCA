
#include "vector.h"
#include <iterator>
#include <fstream>
#include <math.h>
#define __GB(x)   (int)(x / 1024 + 1) , (int)(x < 1024 ? x : 1024)

void writeToFile(float* val , int size , char* fileName){
	std::ofstream ofile(fileName);
	std::ostream_iterator<float> start(ofile , "\n");
	
	std::copy(val , val + size , start);
	ofile.close();
	return;
}

template<class T1>
void copyDevice(T1 a , T1 b , unsigned int c , int d){
	if(d == 0){
		//Host To Device
		cudaMemcpy(a , b , c , cudaMemcpyHostToDevice);
		return;
	}
	if(d == 1){
		//Device To Host
		cudaMemcpy(a , b , c , cudaMemcpyDeviceToHost);
		return;
	}
	else{
		//Device To Device
		cudaMemcpy(a , b , c , cudaMemcpyDeviceToDevice);
		return;
	}
}

__global__ void scaleImage(float* d_face , float sc , int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
		d_face[id] *= sc;
	return;
}

__global__ void computeEigenFace(float* d_face , float* d_eigenVec , float* d_efaces , int sizeVec , int sizeEigen){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id >= sizeVec)
		return;

	float sum = 0;
	for(int i = 0 ; i < sizeEigen ; i ++){
		sum = sum + d_eigenVec[i] * d_face[i * sizeVec + id];
	}

	d_efaces[id] = sum;

	return;
}

__global__ void normalizeVectors(float *d_faceSet , float* d_avgSet , int noFaces , int sizeFaces){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = 0 ; i < sizeFaces ; i ++){
		d_faceSet[id * sizeFaces + i] -= d_avgSet[i];	
	}

	return;
}

__global__ void computeAverage(float *d_faceSet , float* d_avgSet , int noFaces , int sizeFaces){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float average = 0;

	if(id >= sizeFaces)
		return;

	for(int i = 0 ; i < noFaces ; i ++){
		average += d_faceSet[i * sizeFaces + id] / noFaces;
	}
	d_avgSet[id] = average;
	return;
}

__global__ void getDiff(float* f1 , float* f2 , float* diff , int size){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		diff[id] = f1[id] - f2[id];

	return;

}

__global__ void computeProd(float* f1 , float* f2 , float* prod , int sizeVec){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < sizeVec)
		prod[id] = f1[id] * f2[id];
	
	return;
}
 
__global__ void computeTSum(float* vec , float* sum , int size , int start){
	
	int step = 2;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= size)
		return;
	while(step < size){
		
		if(id % step == 0){
			int id1 = id + step / 2;
			if(id1 < size){
				vec[start + id] = vec[start + id] + vec[start + id1];
				
			}
		}
		
		step *= 2;
		__syncthreads();

	}

	
	if(id == 0){
		*sum = vec[start] + vec[start + step / 2];
	}
	return;
}

float getSum(float *data , int size , bool isHost = true){
	
	float* d_data;
	float* d_sum;
	float sum = 0 , sumT;
	if(isHost == true)
		cudaMalloc((void**) &d_data , size * sizeof(float));
	else
		d_data = data;
	cudaMalloc((void**) &d_sum , sizeof(float));
	int maxSize = size / 1024 + 1;
	for(int i = 0 ; i < maxSize ; i ++){
			if(i == maxSize - 1){
				computeTSum<<<1 , size % 1024>>>(d_data , d_sum , size % 1024 , i * 1024);
				copyDevice(&sumT , d_sum , sizeof(float) , 1);
				sum += sumT;
				continue;
			}
			computeTSum<<<1 , 1024>>>(d_data , d_sum , 1024 , i * 1024);
			copyDevice(&sumT , d_sum , sizeof(float) , 1);
			sum += sumT;	
	}

	return sum;
}

void normalizeVec(float* faceSet , int sizeVec , int noVec){
	float* d_faceSet , *d_average;

	cudaMalloc((void**)&d_faceSet , sizeVec * noVec * sizeof(float));
	cudaMalloc((void**)&d_average , sizeVec * sizeof(float));

	cudaMemcpy(d_faceSet , faceSet , sizeVec * noVec * sizeof(float) , cudaMemcpyHostToDevice);
	
	computeAverage<<<__GB(sizeVec)>>>(d_faceSet , d_average , noVec , sizeVec);
	
	normalizeVectors<<<__GB(noVec)>>>(d_faceSet , d_average , noVec , sizeVec);

	cudaMemcpy(faceSet , d_faceSet , sizeVec * noVec * sizeof(float) , cudaMemcpyDeviceToHost);
	return;
}

//Assuming the average is zero
float getCovariance(float* first , float* second , int size){
	float sum;
	float* d_first , *d_second , *d_output;

	float *output;

	cudaMalloc((void**)&d_first , size * sizeof(float));
	cudaMalloc((void**)&d_second , size * sizeof(float));
	cudaMalloc((void**)&d_output , size * sizeof(float));

	cudaMemcpy(d_first , first , size * sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy(d_second , second , size * sizeof(float) , cudaMemcpyHostToDevice);
	
	computeProd<<<__GB(size)>>>(d_first , d_second , d_output , size);
	
	sum = getSum(d_output , size , false);

	cudaFree(d_first);
	cudaFree(d_second);
	cudaFree(d_output);

	return sum / size;
}

float* getCovMat(float* faceSet , int sizeVec , int noVec){
	
	normalizeVec(faceSet , sizeVec , noVec);
	
	float* covMat = new float[noVec * noVec];

	for(int i = 0 ; i < noVec ; i ++){
		for(int j = 0 ; j < noVec ; j ++){
			float* xVec = new float[sizeVec];
			float* yVec = new float[sizeVec];
			std::copy(faceSet + i * sizeVec , faceSet + i * sizeVec + sizeVec , xVec);
			std::copy(faceSet + j * sizeVec , faceSet + j * sizeVec + sizeVec , yVec);
			
			covMat[i * noVec + j] = getCovariance(xVec , yVec , sizeVec);
			delete [] xVec;
			delete [] yVec;
		}
	}

	return covMat;
}

void scaleGrayValUp(float* face , int size){
	float *d_face;

	if(cudaMalloc((void**)&d_face , size * sizeof(float)) != cudaSuccess){
		std::cout<<"Error1";
		_getch();
	}
	if(cudaMemcpy(d_face , face , size * sizeof(float) , cudaMemcpyHostToDevice) != cudaSuccess){
		std::cout<<"Error1";
		_getch();
	}
	
	scaleImage<<<__GB(size)>>>(d_face , 255.0 , size);
	
	if(cudaMemcpy(face , d_face , size * sizeof(float) , cudaMemcpyDeviceToHost) != cudaSuccess){
		std::cout<<"Error while memcpy";
		_getch();
	}

	cudaFree(d_face);
	return;
}



void scaleGrayValDown(float* face , int size){
	float *d_face;

	if(cudaMalloc((void**)&d_face , size * sizeof(float)) != cudaSuccess){
		std::cout<<"Error1";
		_getch();
	}
	if(cudaMemcpy(d_face , face , size * sizeof(float) , cudaMemcpyHostToDevice) != cudaSuccess){
		std::cout<<"Error1";
		_getch();
	}
	

	
	scaleImage<<<__GB(size)>>>(d_face , 1 / 255.0 , size);
	
	if(cudaMemcpy(face , d_face , size * sizeof(float) , cudaMemcpyDeviceToHost) != cudaSuccess){
		std::cout<<"Error while memcpy";
		_getch();
	}


	cudaFree(d_face);
	return;
}

float* returnEigenFaces(float* imageSet , float* eigenMat , int sizeVec , int noEigen , int noSample){
	float* eigenFace = new float[sizeVec * noEigen];
	float *d_eigenX, *d_imageSet , *d_eigenfaceX;
	
	
	cudaMalloc((void**)&d_eigenX , noSample * sizeof(float));
	cudaMalloc((void**)&d_imageSet , noSample * sizeVec * sizeof(float));
	cudaMalloc((void**)&d_eigenfaceX , sizeVec * sizeof(float));
	cudaMemcpy(d_imageSet , imageSet , noSample * sizeVec * sizeof(float) , cudaMemcpyHostToDevice);

	for(int i = 0 ; i < noEigen ; i ++){
		
		cudaMemcpy(d_eigenX , eigenMat + i * noSample , noSample * sizeof(float) , cudaMemcpyHostToDevice);
		
		computeEigenFace<<<__GB(noSample)>>>(d_imageSet , d_eigenX , d_eigenfaceX , sizeVec , noSample);
		
		cudaMemcpy(eigenFace + sizeVec * i , d_eigenfaceX , sizeVec * sizeof(float) , cudaMemcpyDeviceToHost);
	}

	cudaFree(d_imageSet);
	cudaFree(d_eigenX);
	cudaFree(d_eigenfaceX);

	return eigenFace;

}

float* getAverage(float* faceSet , int faceSize , int noSample){
	float *d_faceSet , *d_avg;
	cudaMalloc((void**)&d_faceSet , faceSize * noSample * sizeof(float));
	cudaMalloc((void**)&d_avg , faceSize * sizeof(float));
	
	cudaMemcpy(d_faceSet , faceSet , faceSize * noSample * sizeof(float) , cudaMemcpyDeviceToHost);

	computeAverage<<<__GB(faceSize)>>>(d_faceSet , d_avg , noSample , faceSize);

	float* average = new float[faceSize];

	cudaMemcpy(average , d_avg , faceSize * sizeof(float) , cudaMemcpyDeviceToHost);

	return average;

}

void differenceAverage(float* face , float* avg , int size){
	float* d_face;
	float* d_avg;
	float* d_diff;

	cudaMalloc((void**)&d_face , size * sizeof(float));
	cudaMalloc((void**)&d_avg , size * sizeof(float));
	cudaMalloc((void**)&d_diff , size * sizeof(float));

	cudaMemcpy(d_face , face , size * sizeof(float) , cudaMemcpyDeviceToHost);
	cudaMemcpy(d_avg , avg , size * sizeof(float) , cudaMemcpyDeviceToHost);

	getDiff<<<__GB(size)>>>(d_face , d_avg , d_diff , size);

	cudaMemcpy(face , d_face , size * sizeof(float) , cudaMemcpyDeviceToHost);

	cudaFree(d_avg);
	cudaFree(d_face);
	cudaFree(d_diff);

	return;
}

float* returnEigenWeight(float* eigenface , float* face , int sizeFace , int noEigen){
	
	
	float* eigenWeight = new float[noEigen];

	float* d_eigen , *d_face , *d_prod;

	cudaMalloc((void**)&d_eigen , sizeFace * sizeof(float));
	cudaMalloc((void**)&d_face , sizeFace * sizeof(float));
	cudaMalloc((void**)&d_prod , sizeFace * sizeof(float));
	cudaMemcpy(d_face , face , sizeFace * sizeof(float) , cudaMemcpyHostToDevice);

	for(int i = 0 ; i < noEigen ; i ++){
		cudaMemcpy(d_eigen , eigenface + i * sizeFace , sizeFace * sizeof(float) , cudaMemcpyHostToDevice);
		computeProd<<<__GB(sizeFace)>>>(d_eigen , d_face , d_prod , sizeFace);
		
		/*if(i == 0){
			float* pr = new float[sizeFace];
			copyDevice(pr , d_prod , sizeFace * sizeof(float) , 1);
			writeToFile(eigenface + i * sizeFace , sizeFace , "eigenProd0.txt");
			delete [] pr;
		}
		if(i == 1){
			float* pr = new float[sizeFace];
			copyDevice(pr , d_prod , sizeFace * sizeof(float) , 1);
			writeToFile(eigenface + i * sizeFace , sizeFace , "eigenProd1.txt");
			delete [] pr;
		}
		if(i == 2){
			float* pr = new float[sizeFace];
			copyDevice(pr , d_prod , sizeFace * sizeof(float) , 1);
			writeToFile(eigenface + i * sizeFace , sizeFace , "eigenProd2.txt");
			delete [] pr;
		}*/

		eigenWeight[i] = getSum(d_prod , sizeFace , false) / sqrt((long double)sizeFace);
	}

	cudaFree(d_prod);
	cudaFree(d_eigen);
	cudaFree(d_face);

	return eigenWeight;
}

float* getEigenWtDataSet(float* eigenFace , float* imageSet , int sizeVec , int noEigen , int noSamples){
	float* eigenWeights = new float[noEigen * noSamples];
	//std::cout<<"\n\n\nComputed DataSets\n";
	//char com[50] = "testEigenWt1";
	//char full[60];
	for(int i = 0 ; i < noSamples ; i ++){
		float* eigenWX = returnEigenWeight(eigenFace , imageSet + i * sizeVec , sizeVec , noEigen);
		std::copy(eigenWX , eigenWX + noEigen , eigenWeights + i * noEigen);
		std::cout<<"\nImage "<<i<<" : ";
		//sprintf(full , "%s%d.txt" , com , i);

		for(int j = 0 ; j < noEigen ; j ++)
			std::cout<<eigenWX[j]<<"\t";
		
		//writeToFile(eigenWX , noEigen , full);
		delete[] eigenWX;
	}
	return eigenWeights;
}

float getErrorVal(float* eigenWeight , float* testEigen , int eigenNumber){
	
	float *d_eigenWeight , *d_testEigen , *d_diffEigen , *d_prod;

	cudaMalloc((void**) &d_eigenWeight , eigenNumber * sizeof(float));
	cudaMalloc((void**) &d_testEigen , eigenNumber * sizeof(float));
	cudaMalloc((void**) &d_diffEigen , eigenNumber * sizeof(float));
	cudaMalloc((void**) &d_prod , eigenNumber * sizeof(float));

	cudaMemcpy(d_eigenWeight , eigenWeight , eigenNumber * sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy(d_testEigen , testEigen , eigenNumber * sizeof(float) , cudaMemcpyHostToDevice);

	
	getDiff<<<__GB(eigenNumber)>>>(d_eigenWeight , d_testEigen , d_diffEigen , eigenNumber);

	//float* differ = new float[eigenNumber];
	//copyDevice(differ , d_diffEigen , eigenNumber * sizeof(float) , 1);

	//writeToFile(differ , eigenNumber , "ErrorValues.txt");


	computeProd<<<__GB(eigenNumber)>>>(d_diffEigen , d_diffEigen , d_prod , eigenNumber);
	float sum = getSum(d_prod , eigenNumber , false);
	
	cudaFree(d_eigenWeight);
	cudaFree(d_testEigen);
	cudaFree(d_diffEigen);
	cudaFree(d_prod);

	return sum;

}

float* getAllError(float* facesEigenWt , float* testEigenWt , int eigenNumber , int sampleSize){
	float *error = new float[sampleSize];

	for(int i = 0 ; i < sampleSize ; i ++){
		error[i] = getErrorVal(facesEigenWt + i * eigenNumber , testEigenWt , eigenNumber);
	}

	return error;
}

float* getClassImagePreProcessed(char* filename , int height , int width , float* average){
	float* image;

	char url[100]; sprintf(url , "FaceSet\\Tst\\%s" , filename);
	
	image = returnImageVector(url , width , height);

	scaleGrayValDown(image , width * height);

	float *d_img , *d_average , *d_diff;
	cudaMalloc((void**)&d_img , width * height * sizeof(float));
	cudaMalloc((void**)&d_average , width * height * sizeof(float));
	cudaMalloc((void**)&d_diff , width * height * sizeof(float));
	
	copyDevice(d_img , image , width * height * sizeof(float) , 0);
	copyDevice(d_average , average , width * height * sizeof(float) , 0);
	int size = width * height;
	getDiff<<<__GB(size)>>>(d_img , d_average , d_diff , size);
	copyDevice(image , d_diff , size * sizeof(float) , 1);


	cudaFree(d_img);
	cudaFree(d_average);
	cudaFree(d_diff);

	return image;
}
int main()
{
	 
	//Get Image 
	char dir[] = "FaceSet\\Sets\\";
	int noInput , sc = 1 / 255.0;
	
	
	int height = 20 , width = 20 , noEigen , sizeEachSample = 8;
	
	
	float *imageSet = readImgDir(dir , width , height , noInput);

	//Compute Average;
	scaleGrayValDown(imageSet , noInput * height * width);
	
	noEigen = noInput / 6;

	float *average = getAverage(imageSet , height * width , noInput);
	
	float *covMat = getCovMat(imageSet , width * height , noInput);


	float *eigen = computeEigenValues(covMat , noInput , noEigen);

	writeToFile(eigen , noInput * noEigen , "EigenVal.txt");
	
	float *eFace = returnEigenFaces(imageSet , eigen , width * height , noEigen , noInput);

	
	//EigenFace Constructed
	float *eigenWtDSet = getEigenWtDataSet(eFace , imageSet , height * width , noEigen , noInput);
	writeToFile(eigenWtDSet , noInput * noEigen , "weight2.txt");
	//Compute eigenWeight of the imageSet using created eigenFace

	float* image = getClassImagePreProcessed("test.bmp" , height , width , average);
	float *eW = returnEigenWeight(eFace , image , height * width , noEigen);

	std::cout<<"\n\n\nEigen Weight of Image : ";

	for(int i = 0 ; i < noEigen ; i ++)
		std::cout<<eW[i]<<"\t";

	std::cout<<"\n";

	//Create eigenweight of the image placed at position 0(Zeroth image)
	float *error = getAllError(eigenWtDSet ,  eW , noEigen , noInput);

	for(int i = 0 ; i < noInput ; i ++)
		std::cout<<"Error "<<i<<" : "<< error[i]<<"\n";

	//scaleGrayValUp(eFace , 3 * width * height);
	//scaleGrayValUp(imageSet , width * height);

	//displayImage(imageSet , width , height , 1 , 0);

	_getch();
    return 0;
}
