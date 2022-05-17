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

typedef enum { data_tag, result_tag, terminator_tag } tag;

float real_min = -2.0;
float real_max = 1.0;
float imag_min = -1.5;
float imag_max = 1.5;

#define disp_width 400
#define disp_height 400
#define block_width 10
int N_rectangle = 40;
int arr[disp_height][disp_width];

int p;
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

void calc_block_and_send(int col, int rank) {
  complex c;
  mand m[block_width*disp_height];
  int cnt = 0;
  float scale_real = (real_max - real_min) / disp_width;
  float scale_imag = (imag_max - imag_min) / disp_height;
  printf("[+] calc block(%d~%d) in %d\n", col, col + block_width - 1, rank);
  for (int x = col; x < col + block_width; x++) {
    for (int y = 0; y < disp_height; y++) {
      c.real = real_min + ((float)x * scale_real);
      c.imag = imag_min + ((float)y * scale_imag);
      int color = cal_pixel(c);
      m[cnt].x = x;
      m[cnt].y = y;
      m[cnt].c = color;
      cnt++;
    }
  }
  MPI_Send(&m, cnt, MPI_Mand, 0, result_tag, MPI_COMM_WORLD);
}

  void master() {
    int count = 0, col = 0, slave_rank;
    MPI_Status status;
    mand data[block_width*disp_height];

    for (int i = 1; i < p; i++, col += block_width,
             count++) {  // first assign one block for all slave
      MPI_Send(&col, 1, MPI_INT, i, data_tag, MPI_COMM_WORLD);
    }

    do {
      MPI_Recv(&data, block_width * disp_height, MPI_Mand, MPI_ANY_SOURCE,
               result_tag, MPI_COMM_WORLD,
            &status);
      for (int i = 0; i < block_width * disp_height; i++) display(data[i]);
      count--; /* reduce count as rows received */
      slave_rank = status.MPI_SOURCE;
      if (col < disp_width) {
        MPI_Send(&col, 1, MPI_INT, slave_rank, data_tag, MPI_COMM_WORLD);
        col += block_width;
        count++;
      } else
        MPI_Send(&col, 1, MPI_INT, slave_rank, terminator_tag,
        MPI_COMM_WORLD);
      
    } while (count > 0);  // while remain working slave
    return;
  }

  void slave(int rank) {
    MPI_Status status;
    int col;
    MPI_Recv(&col, 1, MPI_INT, 0, data_tag, MPI_COMM_WORLD, &status);
     while (status.MPI_TAG == data_tag) {
      calc_block_and_send(col, rank);
      MPI_Recv(&col, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    return;
  }

  int main(int argc, char* argv[]) {
    double start, end, duration, global;

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
    printf("Runtime at %d : %f \n", me, duration);

    // calc global duration and save output
    MPI_Reduce(&duration, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (me == 0) {
      printf("Global runtime is %f\n", global);
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