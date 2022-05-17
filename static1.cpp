#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#include <ctime>
#include <sstream>
#include <string>
#include <utility>

#include "mpi.h"

using namespace std;

typedef struct {
  float real;
  float imag;
} complex;

typedef struct {  // mandelbrot (x, y, color)
  int x;
  int y;
  int c;
} mand;

typedef enum { data_tag, terminator_tag } tag;

float real_min = -2.0;
float real_max = 1.0;
float imag_min = -1.5;
float imag_max = 1.5;

#define disp_width 400
#define disp_height 400
int N_rectangle = 40;
int arr[disp_height][disp_width];

int p;
int cols[10];

MPI_Datatype MPI_Mand;

// to output image -
// https://stackoverflow.com/questions/58544166/converting-2d-array-into-a-greyscale-image-in-c
typedef unsigned char U8;

typedef struct {
  U8 p[4];
} color;

void save(const char* file_name, int width, int height) {
  FILE* f = fopen(file_name, "wb");
  color tablo_color[256];
  for (int i = 0; i < 256; i++)
    tablo_color[i] = {(U8)i, (U8)i, (U8)i, (U8)255};  // BGRA 32 bit
  U8 pp[54] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40,
               0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 32};
  *(int*)(pp + 2) = 54 + 4 * width * height;  // file size
  *(int*)(pp + 18) = width;
  *(int*)(pp + 22) = height;
  *(int*)(pp + 42) = height * width * 4;  // bitmap size
  fwrite(pp, 1, 54, f);
  int mx = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      mx = max(mx, arr[i][j]);
      U8 indis = arr[i][j];
      fwrite(tablo_color + indis, 4, 1, f);
    }
  }
  printf("Max color: %d\n", mx);
  fclose(f);
  return;
}
//

int cal_pixel(complex c) {
  int count, max;
  complex z;
  float temp, lengthsq;
  max = 255;
  z.real = 0;
  z.imag = 0;
  count = 0; /* number of iterations */
  do {
    temp = z.real * z.real - z.imag * z.imag + c.real;
    z.imag = 2 * z.real * z.imag + c.imag;
    z.real = temp;
    lengthsq = z.real * z.real + z.imag * z.imag;
    count++;
  } while ((lengthsq < 4.0) && (count < max));
  return count - 1;
}

void display(mand data) {
  arr[data.y][data.x] = data.c;
  return;
}

void calc_col_and_send(int col, int rank) {
  complex c;
  mand m;
  float scale_real = (real_max - real_min) / disp_width;
  float scale_imag = (imag_max - imag_min) / disp_height;
  printf("[+] calc block(%d~%d) in %d\n", col, col+cols[rank]-1, rank);
  for (int x = col; x < col+cols[rank]; x++) {
    for (int y = 0; y < disp_height; y++) {
        c.real = real_min + ((float)x * scale_real);
        c.imag = imag_min + ((float)y * scale_imag);
        int color = cal_pixel(c);
        m.x = x;
        m.y = y;
        m.c = color;
        if (rank == 0) display(m);
        else MPI_Send(&m, 1, MPI_Mand, 0, data_tag, MPI_COMM_WORLD);
     }
  }
}

void master() {
  mand data;
  MPI_Status status;
  for (int i = 1, col = cols[0]; i < p; col = col + cols[i], i++) {
    MPI_Send(&col, 1, MPI_INT, i, data_tag, MPI_COMM_WORLD);
  }
  calc_col_and_send(0, 0);
  for (int i = 0; i < disp_width - cols[0]; i++) {
    for (int j = 0; j < disp_height; j++) {
      MPI_Recv(&data, 1, MPI_Mand, MPI_ANY_SOURCE, data_tag, MPI_COMM_WORLD,
               &status);
      display(data);
    }
  }
  return;
}

void slave(int rank) {
  int col;
  MPI_Status status;
  MPI_Recv(&col, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
           &status);
  calc_col_and_send(col, rank);
  return;
}

int main(int argc, char* argv[]) {
  double start, end, duration, global;
  // clock_t start, end;
  // start = clock();

  // get communicator size and rank of mine
  int me;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  // printf("Size: %d, Rank: %d\n", size, me);

  // define MPI_Mand datatype (three ints)
  MPI_Type_contiguous(3, MPI_INT, &MPI_Mand);
  MPI_Type_commit(&MPI_Mand);

  // set cols (assign blocks to processors)
  int block_width = disp_width / N_rectangle;  // 10
  for (int i = 0; i < p; i++) cols[i] = block_width * (N_rectangle / p);
  //for (int i = 0; i < p; i++) printf("%d : %d\n", i, cols[i]);

  // set timer
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  // run master and slave
  if (me == 0) {
    printf("P: %d\n", p);
    master();
  } else {
    slave(me);
  }

  // end of timer
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  // calc local duration
  duration = end - start;
  //printf("Runtime at %d : %f \n", me, duration);

  // calc global duration and save output
  MPI_Reduce(&duration, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (me == 0) {
    printf("[+] Global runtime : %f sec\n", global);
    ostringstream ss;
    ss << "output-static-" << p << '-' << global << ".bmp";
    string filename = string(ss.str());
    save(filename.c_str(), disp_height, disp_width);
    printf("[+] save %s\n", filename.c_str());
  }

  // finalize
  MPI_Finalize();
  return 0;
}