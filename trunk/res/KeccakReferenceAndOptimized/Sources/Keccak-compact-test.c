/*
Algorithm Name: Keccak
Authors: Guido Bertoni, Joan Daemen, Michaël Peeters and Gilles Van Assche
Implementation by Ronny Van Keer, STMicroelectronics

This code, originally by Ronny Van Keer, is hereby put in the public domain.
It is given as is, without any guarantee.

For more information, feedback or questions, please refer to our website:
http://keccak.noekeon.org/
*/

#include "Keccak-compact.h"

#define cKeccakR_SizeInBytes    (cKeccakR / 8)
#ifndef crypto_hash_BYTES
    #define crypto_hash_BYTES cKeccakR_SizeInBytes
#endif
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#if        (cKeccakB    == 1600) && (cKeccakR == 1024)
    const char *    testVectorFile = "ShortMsgKAT_0.txt";
#elif    (cKeccakB    == 1600) && (cKeccakR == 1152)
    const char *    testVectorFile = "ShortMsgKAT_224.txt";
#elif    (cKeccakB    == 1600) && (cKeccakR == 1088)
    const char *    testVectorFile = "ShortMsgKAT_256.txt";
#elif    (cKeccakB    == 1600) && (cKeccakR == 832)
    const char *    testVectorFile = "ShortMsgKAT_384.txt";
#elif    (cKeccakB    == 1600) && (cKeccakR == 576)
    const char *    testVectorFile = "ShortMsgKAT_512.txt";
#elif    (cKeccakB    == 800) && (cKeccakR == 544)
    const char *    testVectorFile = "ShortMsgKAT_r544c256.txt";
#elif    (cKeccakB    == 400) && (cKeccakR == 144)
    const char *    testVectorFile = "ShortMsgKAT_r144c256.txt";
#elif    (cKeccakB    == 200) && (cKeccakR == 40)
    const char *    testVectorFile = "ShortMsgKAT_r40c160.txt";
#endif

#define    cKeccakMaxMessageSizeInBits		2047
#define    cKeccakMaxMessageSizeInBytes		(cKeccakMaxMessageSizeInBits/8)

#if 	(cKeccakD != 0)
	#define    cKeccakHashRefSizeInBytes		cKeccakD
#elif	(cKeccakB == 1600)
	#define    cKeccakHashRefSizeInBytes		512
#else
	#define    cKeccakHashRefSizeInBytes		cKeccakR_SizeInBytes
#endif


ALIGN unsigned char input[cKeccakMaxMessageSizeInBytes];
ALIGN unsigned char output[cKeccakHashRefSizeInBytes];
ALIGN unsigned char ref[cKeccakHashRefSizeInBytes];

//
// ALLOW TO READ HEXADECIMAL ENTRY (KEYS, DATA, TEXT, etc.)
//
#define MAX_MARKER_LEN      50

int FindMarker(FILE *infile, const char *marker);
int FindMarker(FILE *infile, const char *marker)
{
    char    line[MAX_MARKER_LEN];
    int     i, len;

    len = (int)strlen(marker);
    if ( len > MAX_MARKER_LEN-1 )
        len = MAX_MARKER_LEN-1;

    for ( i=0; i<len; i++ )
        if ( (line[i] = fgetc(infile)) == EOF )
            return 0;
    line[len] = '\0';

    while ( 1 ) {
        if ( !strncmp(line, marker, len) )
            return 1;

        for ( i=0; i<len-1; i++ )
            line[i] = line[i+1];
        if ( (line[len-1] = fgetc(infile)) == EOF )
            return 0;
        line[len] = '\0';
    }

    // shouldn't get here
    return 0;
}

//
// ALLOW TO READ HEXADECIMAL ENTRY (KEYS, DATA, TEXT, etc.)
//
int ReadHex(FILE *infile, BitSequence *A, int Length, char *str);
int ReadHex(FILE *infile, BitSequence *A, int Length, char *str)
{
    int         i, ch, started;
    BitSequence ich;

    if ( Length == 0 ) {
        A[0] = 0x00;
        return 1;
    }
    memset(A, 0x00, Length);
    started = 0;
    i = 0;
    if ( FindMarker(infile, str) )
        while ( (ch = fgetc(infile)) != EOF ) 
        {
            if ( !isxdigit(ch) ) {
                if ( !started ) {
                    if ( ch == '\n' )
                        break;
                    else
                        continue;
                }
                else
                    break;
            }
            started = 1;
            if ( (ch >= '0') && (ch <= '9') )
                ich = ch - '0';
            else if ( (ch >= 'A') && (ch <= 'F') )
                ich = ch - 'A' + 10;
            else if ( (ch >= 'a') && (ch <= 'f') )
                ich = ch - 'a' + 10;

            A[i / 2] = (A[i / 2] << 4) | ich;
            if ( (++i / 2) == Length )
                break;
        }
    else
        return 0;

    return 1;
}

int main( void )
{
    unsigned long long    inlen;
    unsigned long long    offset;
    unsigned long long    size;
    int                    result = 0;
    FILE                *fp_in;
    char                marker[20];
    int                    refLen;
	hashState			state;

    refLen = crypto_hash_BYTES;
    if ((cKeccakD != 0) && (refLen > cKeccakD))
        refLen = cKeccakD;

    printf( "Testing Keccak[r=%u, c=%u, d=%u] using crypto_hash() against %s over %d squeezed bytes\n", cKeccakR, cKeccakB - cKeccakR, cKeccakD, testVectorFile, refLen );
    if ( (fp_in = fopen(testVectorFile, "r")) == NULL ) 
    {
        printf("Couldn't open <%s> for read\n", testVectorFile);
        return 1;
    }

    for ( inlen = 0; inlen <= cKeccakMaxMessageSizeInBytes; ++inlen )
    {
        sprintf( marker, "Len = %u", inlen * 8 );
        if ( !FindMarker(fp_in, marker) )
        {
            printf("ERROR: no test vector found (%u bytes)\n", inlen );
            result = 1;
            break;
        }
        if ( !ReadHex(fp_in, input, (int)inlen, "Msg = ") ) 
        {
            printf("ERROR: unable to read 'Msg' (%u bytes)\n", inlen );
            result = 1;
            break;
        }

        result = crypto_hash( output, input, inlen );
        if ( result != 0 )
        {
            printf("ERROR: crypto_hash() (%u bytes)\n", inlen);
            result = 1;
            break;
        }

        #if (cKeccakD == 0)
        if ( !ReadHex(fp_in, input, refLen, "Squeezed = ") )
        #else
        if ( !ReadHex(fp_in, input, refLen, "MD = ") )
        #endif
        {
            printf("ERROR: unable to read 'Squeezed/MD' (%u bytes)\n", inlen );
            result = 1;
            break;
        }
        if ( memcmp( output, input, refLen ) != 0) 
        {
            printf("ERROR: hash verification (%u bytes)\n", inlen );
            for(result=0; result<refLen; result++)
                printf("%02X ", output[result]);
            printf("\n");
            result = 1;
            break;
        }
    }
    if ( !result )
        printf( "\nSuccess!\n");
	result = 0;

    refLen = cKeccakHashRefSizeInBytes;
    printf( "\nTesting Keccak[r=%u, c=%u, d=%u] using Init/Update/Final() against %s over %d squeezed bytes\n", cKeccakR, cKeccakB - cKeccakR, cKeccakD, testVectorFile, refLen );
    fseek( fp_in, 0, SEEK_SET );
    for ( inlen = 0; inlen <= cKeccakMaxMessageSizeInBits; ++inlen )
    {
        sprintf( marker, "Len = %u", inlen );
        if ( !FindMarker(fp_in, marker) )
        {
            printf("ERROR: no test vector found (%u bits)\n", inlen );
            result = 1;
            break;
        }
        if ( !ReadHex(fp_in, input, (int)inlen, "Msg = ") ) 
        {
            printf("ERROR: unable to read 'Msg' (%u bits)\n", inlen );
            result = 1;
            break;
        }

		result = Init( &state );
        if ( result != 0 )
        {
            printf("ERROR: Init() (%u bits)\n", inlen);
            result = 1;
            break;
        }

		for ( offset = 0; offset < inlen; offset += size )
		{
			//	vary sizes for Update()
			if ( (inlen %8) < 2 )
			{
				//	byte per byte
				size = 8;
			}
			else if ( (inlen %8) < 4 )
			{
				//	incremental
				size = offset + 8;
			}
			else
			{
				//	random
				size = ((rand() % ((inlen + 8) / 8)) + 1) * 8;
			}

			if ( size > (inlen - offset) ) 
			{
				size = inlen - offset;
			}
			//printf("Update() inlen %u, size %u, offset %u\n", (unsigned int)inlen, (unsigned int)size, (unsigned int)offset );
			result = Update( &state, input + offset / 8, size );
	        if ( result != 0 )
	        {
	            printf("ERROR: Update() (%u bits)\n", inlen);
	            result = 1;
	            break;
	        }
		}
		result = Final( &state, output, refLen );
        if ( result != 0 )
        {
            printf("ERROR: Final() (%u bits)\n", inlen);
            result = 1;
            break;
        }

        #if (cKeccakD == 0)
        if ( !ReadHex(fp_in, input, refLen, "Squeezed = ") )
        #else
        if ( !ReadHex(fp_in, input, refLen, "MD = ") )
        #endif
        {
            printf("ERROR: unable to read 'Squeezed/MD' (%u bits)\n", inlen );
            result = 1;
            break;
        }
        if ( memcmp( output, input, refLen ) != 0) 
        {
            printf("ERROR: hash verification (%u bits)\n", inlen );
            for(result=0; result<refLen; result++)
                printf("%02X ", output[result]);
            printf("\n");
            result = 1;
            break;
        }
    }

    fclose( fp_in );
    if ( !result )
        printf( "\nSuccess!\n");

    //printf( "\nPress a key ...");
    //getchar();
    //printf( "\n");
    return ( result );
}



