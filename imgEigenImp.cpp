#include "vector.h"
#include "CImg.h"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "dirent.h"
#include <vector>

using namespace cimg_library;

void displayImage(float* image , int width , int height , int size , int id){
	if (id >= size)
		return;

	CImg<unsigned char>  imageShow(width , height , 1 , 1 , 0);
	for(int i = 0 ; i < width ; i ++){
		for(int j = 0 ; j < height ; j ++){
			imageShow(j , i) = (unsigned char)image[id * width * height + i * height + j];
		}
	}

	CImgDisplay main_disp(imageShow,"Click a point");

	while (!main_disp.is_closed()) {
		main_disp.wait();
    }
	return;
}

float* readImgDir(char* dirName , int width , int height , int& noImage){
	
	DIR* dir = opendir(dirName);

	if(dir == NULL)
		return NULL;
	
	std::vector<char*> fileNames;
	
	int count = 0;
	while (true) {
		struct dirent* ent;
		if((ent = readdir (dir)) == NULL)
			break;
		char* nameI = new char[100];
		strcpy(nameI , ent->d_name);
		if(strlen(nameI) < 5){
			delete[] nameI;
			continue;
		}
		fileNames.push_back(nameI);
		count ++;
	}
	noImage = count;
	float* imageSet = new float[width * height * count];
	
	for(int i = 0 ; i < count ; i ++){
		
		char location[100];
		strcpy(location , dirName);
		strcat(location , fileNames[i]);

		float* image = returnImageVector(location , width , height);
		std::copy(image , image + width * height , imageSet + i * width * height);
		delete [] image;
	}

	return imageSet;

}

float* returnImageSet(char* dirName , int width , int height , int& noSamples){
	return NULL;
}



float* returnImageVector(const char* filename , int width , int height){

	float* faceInfo = new float[width * height];
	CImg<unsigned char> image(filename) , gray(width , height , 1 , 1 , 0);
	image.resize(width , height , 1 , 3 , 5);

	int count = 0;

	cimg_forXY(image , x , y){
		int R = (int)image(x,y,0,0);
		int G = (int)image(x,y,0,1);
		int B = (int)image(x,y,0,2);
		faceInfo[count] = (float)(0.299*R + 0.587*G + 0.114*B);
		count ++;
	}

	//for(int i = 0 ; i < width * height ; i ++){
		//std::cout<<"\n"<<faceInfo[i];
	//}
	return faceInfo;

}

template <class T>
void swap(T& a , T& b){
	T temp;
	temp = a;
	a = b;
	b = temp;
	return;
}

float* computeEigenValues(float* matrix , int size , int noEigen){
	Eigen::MatrixXd mat(size , size);
	float* eigenVec = new float[size * noEigen];

	for(int i = 0 ; i < size ; i ++)
		for(int j = 0 ; j < size ; j ++){
			mat(i , j) = matrix[i * size + j];
		}



	Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(mat);
	
	Eigen::MatrixXd evec = eigenSolver.eigenvectors().real();
	Eigen::MatrixXd eval = eigenSolver.eigenvalues().real();

	float *eigenVal;
	int *eigenIndex;
	eigenVal = new float[size];
	eigenIndex = new int[size];

	for(int i = 0 ; i < size ; i ++){
		eigenVal[i] = eval(i , 0);
		eigenIndex[i] = i;
	}

	for(int i = 0 ; i < size - 1 ; i ++){
		for(int j = i + 1 ; j < size ; j ++){
			if(abs(eigenVal[i]) < abs(eigenVal[j])){
				
				swap(eigenVal[i] , eigenVal[j]);
				swap(eigenIndex[i] , eigenIndex[j]);
			}
		}
	}

	for(int i = 0 ; i < noEigen ; i ++){
		for(int j = 0 ; j < size ; j ++){
			eigenVec[i * size + j] = evec(j , eigenIndex[i]);
		}
	}

	return eigenVec;

}