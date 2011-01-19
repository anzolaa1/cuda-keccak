/*
Algorithm Name: Keccak
Authors: Guido Bertoni, Joan Daemen, Michaël Peeters and Gilles Van Assche

This code, originally by Guido Bertoni, Joan Daemen, Michaël Peeters and
Gilles Van Assche as a part of the SHA-3 submission, is hereby put in the
public domain. It is given as is, without any guarantee.

For more information, feedback or questions, please refer to our website:
http://keccak.noekeon.org/
*/

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "displayIntermediateValues.h"
#include "KeccakNISTInterface.h"
#include "KeccakF-1600-interface.h"
#include "KeccakF-1600-reference.h"

int genKAT_main();

void displayPermutationIntermediateValues()
{
    unsigned char state[KeccakPermutationSizeInBytes];
    const char *fileName = "KeccakPermutationIntermediateValues.txt";
    FILE *f;

    f = fopen(fileName, "w");
    if (f == NULL)
        printf("Could not open %s\n", fileName);
    else {
        KeccakInitialize();
        fprintf(f, "+++ The round constants +++\n");
        fprintf(f, "\n");
        displayRoundConstants(f);

        fprintf(f, "+++ The rho offsets +++\n");
        fprintf(f, "\n");
        displayRhoOffsets(f);

        displaySetIntermediateValueFile(f);
        displaySetLevel(3);

        fprintf(f, "+++ Example with the all-zero input +++\n");
        fprintf(f, "\n");
        memset(state, 0, KeccakPermutationSizeInBytes);
        KeccakPermutation(state);

        fprintf(f, "+++ Example taking the previous output as input +++\n");
        fprintf(f, "\n");
        KeccakPermutation(state);

        fclose(f);
        displaySetIntermediateValueFile(0);
    }
}

void alignLastByteOnLSB(const unsigned char *in, unsigned char *out, unsigned int length)
{
    unsigned int lengthInBytes;

    lengthInBytes = (length+7)/8;
    memcpy(out, in, lengthInBytes);
    if ((length % 8) != 0)
        out[lengthInBytes-1] = out[lengthInBytes-1] >> (8-(length%8));
}

void displayMessageInInternalConvention(const unsigned char *message, unsigned int length)
{
    unsigned char *messageInternal;

    messageInternal = malloc((length+7)/8);
    alignLastByteOnLSB(message, messageInternal, length);
    displayBits(2, "Input message (in bits)", message, length, 1);
    displayBits(2, "Input message (in bits, after the formal bit reordering)", messageInternal, length, 0);
    displayBytes(2, "Input message (last byte aligned on LSB)", messageInternal, (length+7)/8);
    free(messageInternal);
}

void displaySpongeIntermediateValuesOne(const unsigned char *message, unsigned int messageLength, unsigned int digestSize)
{
    hashState state;
    unsigned char digest[512];

    displayBytes(1, "Input message (last byte aligned on MSB)", message, (messageLength+7)/8);

    displayMessageInInternalConvention(message, messageLength);

    Init(&state, digestSize);
    displayStateAsBytes(1, "Initial state", state.state);
    Update(&state, message, messageLength);
    Final(&state, digest);
    if (digestSize == 0) {
        displayText(1, "--- Switching to squeezing phase ---");
        Squeeze(&state, digest, 4096);
    }
}

void displaySpongeIntermediateValuesFew(FILE *f, unsigned int digestSize)
{
    const unsigned char *message1 = "\x53\x58\x7B\xC8";
    unsigned int message1Length = 29;
    const unsigned char *message2 = 
        "\x83\xAF\x34\x27\x9C\xCB\x54\x30\xFE\xBE\xC0\x7A\x81\x95\x0D\x30"
        "\xF4\xB6\x6F\x48\x48\x26\xAF\xEE\x74\x56\xF0\x07\x1A\x51\xE1\xBB"
        "\xC5\x55\x70\xB5\xCC\x7E\xC6\xF9\x30\x9C\x17\xBF\x5B\xEF\xDD\x7C"
        "\x6B\xA6\xE9\x68\xCF\x21\x8A\x2B\x34\xBD\x5C\xF9\x27\xAB\x84\x6E"
        "\x38\xA4\x0B\xBD\x81\x75\x9E\x9E\x33\x38\x10\x16\xA7\x55\xF6\x99"
        "\xDF\x35\xD6\x60\x00\x7B\x5E\xAD\xF2\x92\xFE\xEF\xB7\x35\x20\x7E"
        "\xBF\x70\xB5\xBD\x17\x83\x4F\x7B\xFA\x0E\x16\xCB\x21\x9A\xD4\xAF"
        "\x52\x4A\xB1\xEA\x37\x33\x4A\xA6\x64\x35\xE5\xD3\x97\xFC\x0A\x06"
        "\x5C\x41\x1E\xBB\xCE\x32\xC2\x40\xB9\x04\x76\xD3\x07\xCE\x80\x2E"
        "\xC8\x2C\x1C\x49\xBC\x1B\xEC\x48\xC0\x67\x5E\xC2\xA6\xC6\xF3\xED"
        "\x3E\x5B\x74\x1D\x13\x43\x70\x95\x70\x7C\x56\x5E\x10\xD8\xA2\x0B"
        "\x8C\x20\x46\x8F\xF9\x51\x4F\xCF\x31\xB4\x24\x9C\xD8\x2D\xCE\xE5"
        "\x8C\x0A\x2A\xF5\x38\xB2\x91\xA8\x7E\x33\x90\xD7\x37\x19\x1A\x07"
        "\x48\x4A\x5D\x3F\x3F\xB8\xC8\xF1\x5C\xE0\x56\xE5\xE5\xF8\xFE\xBE"
        "\x5E\x1F\xB5\x9D\x67\x40\x98\x0A\xA0\x6C\xA8\xA0\xC2\x0F\x57\x12"
        "\xB4\xCD\xE5\xD0\x32\xE9\x2A\xB8\x9F\x0A\xE1";
    unsigned int message2Length = 2008;

    fprintf(f, "+++ Example with a small message +++\n");
    fprintf(f, "\n");
    fprintf(f, "This is the message of length 29 from ShortMsgKAT.txt.\n");
    fprintf(f, "\n");
    displaySpongeIntermediateValuesOne(message1, message1Length, digestSize);

    fprintf(f, "+++ Example with a larger message +++\n");
    fprintf(f, "\n");
    fprintf(f, "This is the message of length 2008 from ShortMsgKAT.txt.\n");
    fprintf(f, "\n");
    displaySpongeIntermediateValuesOne(message2, message2Length, digestSize);
}

void displaySpongeIntermediateValues()
{
    const unsigned int digestSize[5] = {0, 224, 256, 384, 512};
    char fileName[256];
    FILE *f;
    unsigned int i;

    for(i=0; i<5; i++) {
        sprintf(fileName, "KeccakSpongeIntermediateValues_%d.txt", digestSize[i]);
        f = fopen(fileName, "w");
        if (f == NULL)
            printf("Could not open %s\n", fileName);
        else {
            displaySetIntermediateValueFile(f);
            displaySetLevel(2);

            displaySpongeIntermediateValuesFew(f, digestSize[i]);

            fclose(f);
            displaySetIntermediateValueFile(0);
        }
    }
}

#define refLenMax 128

void displayTest2040(unsigned int rate, unsigned int capacity, unsigned char diversifier)
{
    const char  testVectorMessage[] =
        "\x3A\x3A\x81\x9C\x48\xEF\xDE\x2A\xD9\x14\xFB\xF0\x0E\x18\xAB\x6B"
        "\xC4\xF1\x45\x13\xAB\x27\xD0\xC1\x78\xA1\x88\xB6\x14\x31\xE7\xF5"
        "\x62\x3C\xB6\x6B\x23\x34\x67\x75\xD3\x86\xB5\x0E\x98\x2C\x49\x3A"
        "\xDB\xBF\xC5\x4B\x9A\x3C\xD3\x83\x38\x23\x36\xA1\xA0\xB2\x15\x0A"
        "\x15\x35\x8F\x33\x6D\x03\xAE\x18\xF6\x66\xC7\x57\x3D\x55\xC4\xFD"
        "\x18\x1C\x29\xE6\xCC\xFD\xE6\x3E\xA3\x5F\x0A\xDF\x58\x85\xCF\xC0"
        "\xA3\xD8\x4A\x2B\x2E\x4D\xD2\x44\x96\xDB\x78\x9E\x66\x31\x70\xCE"
        "\xF7\x47\x98\xAA\x1B\xBC\xD4\x57\x4E\xA0\xBB\xA4\x04\x89\xD7\x64"
        "\xB2\xF8\x3A\xAD\xC6\x6B\x14\x8B\x4A\x0C\xD9\x52\x46\xC1\x27\xD5"
        "\x87\x1C\x4F\x11\x41\x86\x90\xA5\xDD\xF0\x12\x46\xA0\xC8\x0A\x43"
        "\xC7\x00\x88\xB6\x18\x36\x39\xDC\xFD\xA4\x12\x5B\xD1\x13\xA8\xF4"
        "\x9E\xE2\x3E\xD3\x06\xFA\xAC\x57\x6C\x3F\xB0\xC1\xE2\x56\x67\x1D"
        "\x81\x7F\xC2\x53\x4A\x52\xF5\xB4\x39\xF7\x2E\x42\x4D\xE3\x76\xF4"
        "\xC5\x65\xCC\xA8\x23\x07\xDD\x9E\xF7\x6D\xA5\xB7\xC4\xEB\x7E\x08"
        "\x51\x72\xE3\x28\x80\x7C\x02\xD0\x11\xFF\xBF\x33\x78\x53\x78\xD7"
        "\x9D\xC2\x66\xF6\xA5\xBE\x6B\xB0\xE4\xA9\x2E\xCE\xEB\xAE\xB1";
    const int refLen = refLenMax;
    unsigned char output[refLenMax];
    unsigned int offset;
    hashState state;

    InitEx( &state, rate, capacity, diversifier );
    Update( &state, testVectorMessage, 2040 );
    Final( &state, 0 );
    Squeeze( &state, output, refLen*8 );
    printf("Message of size 2040 bits with Keccak[r=%d, c=%d, d=%d]\n", rate, capacity, (int)diversifier);
    for(offset=0; offset<refLen; offset++)
        printf("\\x%02X", output[offset]);
    printf("\n\n");
}

void displayTests2040()
{
    displayTest2040(1152,  448, 28);
    displayTest2040(1088,  512, 32);
    displayTest2040(1024,  576,  0);
    displayTest2040( 832,  768, 48);
    displayTest2040( 576, 1024, 64);
}

void displayAllInOneTest(unsigned int rate, unsigned int capacity, unsigned char diversifier)
{
    #define refLenMax 128
    const int refLen = refLenMax;
    unsigned char input[512];
    unsigned char output[refLenMax];
    unsigned char ref[refLen];
    unsigned int inlen, offset, size;
    int result;
    hashState state;

    // Acumulated test vector for crypto_hash()
    memset( ref, 0x00, sizeof(ref) );

    for ( inlen = 0; inlen <= 4096;
        (inlen < 2*8) ? inlen++ : ((inlen < 32*8) ? (inlen += 8) : (inlen <<= 1)) ) {
        unsigned int i;
        unsigned int bytesize = (unsigned int)((inlen + 7) / 8);

        for ( i = 0; i < bytesize; ++i )
            input[i] = (unsigned char)(i - bytesize);

        result = InitEx( &state, rate, capacity, diversifier );

        for ( offset = 0; offset < inlen; offset += size ) {
            // vary sizes for Update()
            if ( (inlen %8) < 2 )
                // byte per byte
                size = 8;
            else if ( (inlen %8) < 4 )
                // incremental
                size = offset + 8;
            else
                // random
                size = ((rand() % ((inlen + 8) / 8)) + 1) * 8;

            if ( size > (inlen - offset) )
                size = inlen - offset;

            result = Update( &state, input + offset / 8, size );
        }
        result = Final( &state, 0 );
        result = Squeeze( &state, output, refLen*8 );

        for ( i = 0; i < (unsigned int)refLen; ++i )
            ref[i] ^= output[i];
    }
    printf("All-in-one test for Keccak[r=%d, c=%d, d=%d]\n", rate, capacity, (int)diversifier);
    for(offset=0; offset<refLen; offset++)
        printf("\\x%02X", ref[offset]);
    printf("\n\n");
}

void displayAllInOneTests()
{
    displayAllInOneTest(1152,  448, 28);
    displayAllInOneTest(1088,  512, 32);
    displayAllInOneTest(1024,  576,  0);
    displayAllInOneTest( 832,  768, 48);
    displayAllInOneTest( 576, 1024, 64);
}

int main()
{
    displayPermutationIntermediateValues();
    displaySpongeIntermediateValues();
    return genKAT_main();
}
