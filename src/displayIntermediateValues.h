/*
Algorithm Name: Keccak
Authors: Guido Bertoni, Joan Daemen, Michaël Peeters and Gilles Van Assche

This code, originally by Guido Bertoni, Joan Daemen, Michaël Peeters and
Gilles Van Assche as a part of the SHA-3 submission, is hereby put in the
public domain. It is given as is, without any guarantee.

For more information, feedback or questions, please refer to our website:
http://keccak.noekeon.org/
*/

#ifndef _displayIntermediateValues_h_
#define _displayIntermediateValues_h_

#include <stdio.h>
#include "costants.h"

void displaySetIntermediateValueFile(FILE *f);
void displaySetLevel(int level);
void displayBytes(int level, const char *text, const unsigned char *bytes, unsigned int size);
//void displayBits(int level, const char *text, const unsigned char *data, unsigned int size, int MSBfirst);
void displayStateAsBytes(int level, const char *text, const unsigned char *state);
void displayStateAsWords(int level, const char *text, const unsigned long long int *state);
void displayRoundNumber(int level, unsigned int i);
void displayText(int level, const char *text);

#endif
