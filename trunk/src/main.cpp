#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cutil_inline.h>

#define ind(x, y) (((x)%5)+5*((y)%5))


#include "kernel.h"
#include "costants.h"
extern "C"{
	#include "KeccakF1600reference.h"
	#include "displayIntermediateValues.h"
}
int threadsNumber = -1;
int messageLenght = -1;
bool debug = false;
bool cpu = false;
unsigned long long *hash;
unsigned int timer;

void printUsage();
int startKernel();
int startCpu();
void InitializeRhoOffsets(unsigned int* array);
void InitRoundConstants(unsigned long long* array);
int LFSR(unsigned char *LFSR);


int main(int argc, char** argv){
	
	if(argc==1){ //no param -> print help
		printUsage();
		return 0;
	}
	
	for(int i=1; i < argc; i++){
		if(strcmp(argv[i],"-n")==0 && i+1 <argc){ //number of thread (and of messages too)
				threadsNumber = atoi(argv[i+1]);
				i++;
				if(threadsNumber ==0){
					std::cout << std::endl;
					std::cout << "INVALID PARAM FOR THREAD NUMBER" << std::endl;
					printUsage();
					return -1;
				}
				continue;
				
		}
		if(strcmp(argv[i],"-l")==0 && i+1 <argc){ //messages's lenght
				messageLenght = atoi(argv[i+1]);
				i++;
				if(messageLenght == 0){
					std::cout << std::endl;
					std::cout << "INVALID PARAM FOR MESSAGE LENGHT" << std::endl;
					printUsage();
					return -1;
				}
				continue;
			 
		}
		if(strcmp(argv[i],"-h")==0){ //help
			printUsage();
			return 0;
		}
		
		if(strcmp(argv[i],"-d")==0){ //debug mode
			debug = true;
			continue;
		}
		
		if(strcmp(argv[i],"-c")==0){ //debug mode
			cpu = true;
			continue;
		}
		
		//a param is not reconized: error -> print usage
		std::cout << std::endl;
		std::cout << "Unreconized param " << argv[i] << std::endl;
		printUsage();
		return -1;
	}
		
	if(messageLenght < 0){
		std::cout << "INVALID PARAM FOR MESSAGE LENGHT" << std::endl;
		printUsage();
		return -1;
	}
	
	if(threadsNumber < 0){
		std::cout << "INVALID PARAM FOR THREAD NUMBER" << std::endl;
		printUsage();
		return -1;
	}
	return startKernel();
}

unsigned long long roundConstant[24];
unsigned int rhoOffsets[25];


int startKernel(){
	if(!cpu){
		//TODO init round and rhoOffset
		InitializeRhoOffsets(rhoOffsets);
		InitRoundConstants(roundConstant);
		init_cuda(threadsNumber,roundConstant,rhoOffsets); 
		alloc_memory();
	}else{
		cutilCheckError(cutCreateTimer(&timer));
		cutilCheckError(cutStartTimer(timer));
		KeccakInitialize();
		FILE* f = fopen("intermediateResultCpu.txt","w");
		if(f)
			displaySetIntermediateValueFile(f);
		else
			std::cout << "ERROR OPENING OUTPUT FILE" <<std::endl;
	}
	
	//allocate memory for the hash
	hash = (unsigned long long*)malloc(25*threadsNumber*sizeof(unsigned long long));
	if(cpu){
		KeccakInitializeState((unsigned char*)hash);
	}
	unsigned long long *messages;
	srand( time(NULL) );	
	//initializaiotn of the ptr array to 0
	messages = NULL;
	
	for(int l=0;l<messageLenght;l++){
	
		if(l!=0){ //if a kernel is still running we have to wait its termination
			//TODO whait the termination of the kernel
		}
		
		//memory allocation for new messages
		try{
			
			if(l!=0){ //if we are not processing the head of the messages we can free the old segments
				if(messages!=NULL){
					free(messages);
					messages = NULL;
				}else{				
					throw std::string("deallocation error, attempting to free a NULL pointer (may be it was free before and setted to NULL?)");
				}
			}
			messages = (unsigned long long*)malloc(25*threadsNumber*sizeof( unsigned long long ));
			if(messages == NULL)
				throw std::string("allocation error, malloc returned a NULL pointer");
			
		
		}catch(std::string e){
			std::cerr << "An exception was raised during the message's memory deallocation/reallocation:" << std::endl << e << std::endl;
			return -1;
		}
		
		
		//random initialization of the new messages	(in debug mode all is initialize to zero)
		if(!debug){
			//std::cout << "Initialization of the messages with random values" << std::endl;
			for(unsigned int i=0;i<threadsNumber*25;i++){
					messages[i] = rand();
			}
		}else{
			//std::cout << "Initialization of the messages with all zeros" << std::endl;			
			memset(messages,0,25*threadsNumber*sizeof(unsigned long long));
		}
		if(!cpu){
			launch_kernel(messages,l); //launch the kernel
			//std::cout << "Kernel launched" << std::endl;
		}else{
			//std::cout << "Absorb number: " << l << std::endl;
			KeccakAbsorb((unsigned char *)hash, (unsigned char *)messages, 25);
		}
	}

	
	if(!cpu){
		//retrival of the computed hashes
		get_state(hash);
	
		//free the gpu memory
		free_memory();
	}else{
		cutilCheckError(cutStopTimer(timer));
		float milliseconds = cutGetTimerValue(timer);
		cutilCheckError(cutDeleteTimer(timer));
		std::cout << "Tempo totale: ";
		printf("%f ms\n",milliseconds);
	}
	
	//print the hashes
	//std::cout << "Hashes:" << std::endl;
	
	/*for(unsigned int n=0; n< threadsNumber; n++)
		for(unsigned int i=0;i<25;i++){
			std::cout << hash[i+25*n] << std::endl;
		}*/
	
	
	
	return 0;
}



void InitializeRhoOffsets(unsigned int* array)
{
    unsigned int x, y, t, newX, newY;

    array[ind(0, 0)] = 0;
    x = 1;
    y = 0;
    for(t=0; t<24; t++) {
        array[ind(x, y)] = ((t+1)*(t+2)/2) % 64;
        newX = (0*x+1*y) % 5;
        newY = (2*x+3*y) % 5;
        x = newX;
        y = newY;
    }
}


void InitRoundConstants(unsigned long long* array)
{
    unsigned char LFSRstate = 0x01;
    unsigned int i, j, bitPosition;

    for(i=0; i<24; i++) {
        array[i] = 0;
        for(j=0; j<7; j++) {
            bitPosition = (1<<j)-1; //2^j-1
            if (LFSR(&LFSRstate))
                array[i] ^= (unsigned long long)1<<bitPosition;
        }
    }
}

int LFSR(unsigned char *LFSR)
{
    int result = ((*LFSR) & 0x01) != 0;
    if (((*LFSR) & 0x80) != 0)
        // Primitive polynomial over GF(2): x^8+x^6+x^5+x^4+1
        (*LFSR) = ((*LFSR) << 1) ^ 0x71;
    else
        (*LFSR) <<= 1;
    return result;
}




using namespace std;

void printUsage(){
	cout << endl;
	cout << "USAGE OPTIONS:" << endl;
	cout << "h:		print this message" << endl;
	cout << "n:		number of threads to start" << endl;
	cout << "l:		lenght of the messages to process (lenght = l*64*25 bit)" << endl;
	cout << "c:		run in cpu mode" << endl;
	cout << "d:		run in debug mode, in debug mode all the messages\n		are set to a series of zeros" << endl;
	cout << endl;
}
	

