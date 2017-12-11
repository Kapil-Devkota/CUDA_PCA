
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <conio.h>

float* readImgDir(char* dirname , int width , int height , int& noInput);

float* returnImageVector(const char* filename , int width , int height);


float* computeEigenValues(float* , int , int);

void displayImage(float* , int width , int height , int size , int id);