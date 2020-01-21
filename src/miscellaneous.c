#ifdef CUDA_OPT
#include "main.h"

void field_saver( void* phi, int length, char* datatype, char* filename ){
  int i;
  FILE *f;

  f = fopen(filename, "w");
  if(f == NULL){
    printf("Error opening file!\n");
    return;
  }

  for(i=0; i<length; i++){

    if(strcmp(datatype, "float")){
      float buf = ((float*)phi)[i];
      fprintf(f, "[%d]-th entry = %f+i%f\n", i, creal(buf), cimag(buf));
    }
    else{
      double buf = ((double*)phi)[i];
      fprintf(f, "[%d]-th entry = %lf+i%lf\n", i, creal(buf), cimag(buf));
    }

  }

  fclose(f);
}
#endif
