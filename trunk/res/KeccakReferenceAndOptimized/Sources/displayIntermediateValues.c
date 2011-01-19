/*
Algorithm Name: Keccak
Authors: Guido Bertoni, Joan Daemen, Michaël Peeters and Gilles Van Assche

This code, originally by Guido Bertoni, Joan Daemen, Michaël Peeters and
Gilles Van Assche as a part of the SHA-3 submission, is hereby put in the
public domain. It is given as is, without any guarantee.

For more information, feedback or questions, please refer to our website:
http://keccak.noekeon.org/
*/

#include <stdio.h>
#include "displayIntermediateValues.h"
#include "KeccakNISTInterface.h"

FILE *intermediateValueFile = 0;
int displayLevel = 0;

void displaySetIntermediateValueFile(FILE *f)
{
    intermediateValueFile = f;
}

void displaySetLevel(int level)
{
    displayLevel = level;
}

void displayBytes(int level, const char *text, const unsigned char *bytes, unsigned int size)
{
    unsigned int i;

    if ((intermediateValueFile) && (level <= displayLevel)) {
        fprintf(intermediateValueFile, "%s:\n", text);
        for(i=0; i<size; i++)
            fprintf(intermediateValueFile, "%02X ", bytes[i]);
        fprintf(intermediateValueFile, "\n");
        fprintf(intermediateValueFile, "\n");
    }
}

void displayBits(int level, const char *text, const unsigned char *data, unsigned int size, int MSBfirst)
{
    unsigned int i, iByte, iBit;

    if ((intermediateValueFile) && (level <= displayLevel)) {
        fprintf(intermediateValueFile, "%s:\n", text);
        for(i=0; i<size; i++) {
            iByte = i/8;
            iBit = i%8;
            if (MSBfirst)
                fprintf(intermediateValueFile, "%d ", ((data[iByte] << iBit) & 0x80) != 0);
            else
                fprintf(intermediateValueFile, "%d ", ((data[iByte] >> iBit) & 0x01) != 0);
        }
        fprintf(intermediateValueFile, "\n");
        fprintf(intermediateValueFile, "\n");
    }
}

void displayStateAsBytes(int level, const char *text, const unsigned char *state)
{
    displayBytes(level, text, state, KeccakPermutationSizeInBytes);
}

void displayStateAsWords(int level, const char *text, const unsigned long long int *state)
{
    unsigned int i;

    if ((intermediateValueFile) && (level <= displayLevel)) {
        fprintf(intermediateValueFile, "%s:\n", text);
        for(i=0; i<KeccakPermutationSize/64; i++) {
            fprintf(intermediateValueFile, "%08X", (unsigned int)(state[i] >> 32));
            fprintf(intermediateValueFile, "%08X", (unsigned int)(state[i] & 0xFFFFFFFFULL));
            if ((i%5) == 4)
                fprintf(intermediateValueFile, "\n");
            else
                fprintf(intermediateValueFile, " ");
        }
    }
}

void displayRoundNumber(int level, unsigned int i)
{
    if ((intermediateValueFile) && (level <= displayLevel)) {
        fprintf(intermediateValueFile, "\n");
        fprintf(intermediateValueFile, "--- Round %d ---\n", i);
        fprintf(intermediateValueFile, "\n");
    }
}

void displayText(int level, const char *text)
{
    if ((intermediateValueFile) && (level <= displayLevel)) {
        fprintf(intermediateValueFile, text);
        fprintf(intermediateValueFile, "\n");
        fprintf(intermediateValueFile, "\n");
    }
}
