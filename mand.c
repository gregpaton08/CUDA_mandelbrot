/* Sequential Mandelbrot program */

#include <stdio.h>
#include <string.h>
#include <math.h>

#define		X_RESN	800       /* x resolution */
#define		Y_RESN	800       /* y resolution */

typedef struct complextype
	{
        float real, imag;
	} Compl;


int main ()
{
	unsigned int        width, height,                  /* window size */
                        x, y,                           /* window position */
                        border_width,                   /*border width in pixels */
                        display_width, display_height,  /* size of screen */
                        screen;                         /* which screen */

	char       *window_name = "Mandelbrot Set", *display_name = NULL;
	unsigned
	long		valuemask = 0;
	FILE		*fp, *fopen ();
	char		str[100];
	

   /* Mandlebrot variables */
    int i, j, k;
    Compl	z, c;
    float	lengthsq, temp;
    
    
      	 
    /* Calculate and draw points */

    for(i = 0; i < X_RESN; i++) {
        for(j = 0; j < Y_RESN; j++) {

            z.real = z.imag = 0.0;
            c.real = ((float) j - 400.0)/200.0;               /* scale factors for 800 x 800 window */
            c.imag = ((float) i - 400.0)/200.0;
            k = 0;

            do  {                                             /* iterate for pixel color */

                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2.0*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real+z.imag*z.imag;
                k++;

            } while (lengthsq < 4.0 && k < 100);
        }
    }
    
	sleep (30);

	/* Program Finished */

}

